# server.py
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import re
import openai
import asyncio
from contextlib import asynccontextmanager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# -----------------------------
# Configuration / clients
# -----------------------------
# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# OpenAI API key
openai_api_key = os.getenv("EMERGENT_LLM_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OpenAI API key not found in EMERGENT_LLM_KEY or OPENAI_API_KEY")

openai.api_key = openai_api_key

# Create uploads directory
UPLOADS_DIR = ROOT_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)

# Lifespan to close DB client on shutdown (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        client.close()

app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# -----------------------------
# Models
# -----------------------------
class TeamMember(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str
    skills: List[str] = Field(default_factory=list)
    isActive: bool = True
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TeamMemberCreate(BaseModel):
    name: str
    role: str
    skills: List[str] = Field(default_factory=list)

class TeamMemberUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    skills: Optional[List[str]] = None
    isActive: Optional[bool] = None

class Meeting(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    date: datetime
    audioUrl: str
    transcript: Optional[str] = None
    status: str = "uploaded"  # uploaded | transcribing | transcribed | processing | completed | error
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MeetingSummary(BaseModel):
    id: str
    title: str
    date: datetime
    status: str
    taskCount: int = 0

class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    meetingId: str
    description: str
    assignedToMemberId: Optional[str] = None
    assignedToName: Optional[str] = None
    priority: str = "Medium"  # Critical | High | Medium | Low
    deadlineOriginal: Optional[str] = None
    deadlineDate: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    blocking: bool = False
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TaskUpdate(BaseModel):
    description: Optional[str] = None
    assignedToMemberId: Optional[str] = None
    priority: Optional[str] = None
    deadlineDate: Optional[datetime] = None
    deadlineOriginal: Optional[str] = None
    reasoning: Optional[str] = None
    dependencies: Optional[List[str]] = None

# -----------------------------
# Task extraction logic (unchanged, kept same behavior)
# -----------------------------
class TaskExtractor:
    def __init__(self, team_members: List[TeamMember], meeting_date: datetime):
        self.team_members = team_members
        self.meeting_date = meeting_date
        
        # Task indicators
        self.obligation_phrases = [
            r'\bneed to\b', r'\bwe need to\b', r'\bwe need someone to\b',
            r'\bshould\b', r'\bhave to\b', r'\bmust\b', r"\blet's\b",
            r'\bwe should\b', r'\bplan this\b', r'\bsomeone should\b'
        ]
        
        self.action_verbs = [
            'fix', 'update', 'design', 'write', 'implement', 'test',
            'optimize', 'improve', 'document', 'refactor', 'create',
            'build', 'develop', 'review', 'deploy', 'configure'
        ]
        
        # Priority keywords
        self.critical_keywords = ['critical', 'blocking', 'blocker', 'urgent', 'must', 'asap']
        self.high_keywords = ['high priority', 'important', 'before release', 'soon']
        self.low_keywords = ['can wait', 'next sprint', 'later', 'low priority']
        
        # Deadline patterns
        self.deadline_patterns = [
            (r'\btomorrow\b', 1),
            (r'\bby friday\b|\bbefore friday\b', None),
            (r'\bend of (this )?week\b', None),
            (r'\bnext monday\b', None),
            (r'\bnext week\b', 7),
            (r'\bwednesday\b', None),
        ]
        
    def extract_tasks(self, transcript: str) -> List[dict]:
        sentences = self._split_into_sentences(transcript)
        task_sentences = []
        
        for sentence in sentences:
            if self._is_task_sentence(sentence):
                task_sentences.append(sentence)
        
        tasks = []
        for sentence in task_sentences:
            task_data = self._extract_task_details(sentence)
            if task_data:
                tasks.append(task_data)
        
        self._extract_dependencies(tasks)
        
        return tasks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.?!\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_task_sentence(self, sentence: str) -> bool:
        sentence_lower = sentence.lower()
        has_obligation = any(re.search(pattern, sentence_lower) for pattern in self.obligation_phrases)
        has_action = any(verb in sentence_lower for verb in self.action_verbs)
        return has_obligation or has_action
    
    def _extract_task_details(self, sentence: str) -> Optional[dict]:
        description = self._extract_description(sentence)
        if not description:
            return None
        
        assigned_member_id, assigned_name = self._extract_assignee(sentence)
        
        reasoning = None
        if not assigned_member_id:
            assigned_member_id, assigned_name, reasoning = self._auto_assign(description)
        else:
            reasoning = f"Explicitly mentioned {assigned_name} in the meeting."
        
        priority = self._extract_priority(sentence)
        deadline_original, deadline_date = self._extract_deadline(sentence)
        blocking = self._is_blocking(sentence)
        
        return {
            'description': description,
            'assignedToMemberId': assigned_member_id,
            'assignedToName': assigned_name,
            'priority': priority,
            'deadlineOriginal': deadline_original,
            'deadlineDate': deadline_date.isoformat() if deadline_date else None,
            'reasoning': reasoning,
            'blocking': blocking,
            'dependencies': [],
            'originalSentence': sentence
        }
    
    def _extract_description(self, sentence: str) -> str:
        description = sentence
        fillers = [
            r'\bwe need to\b',
            r'\bwe need someone to\b',
            r'\bwe should\b',
            r"\blet's\b",
            r'\bsomeone should\b',
            r'\bI think\b',
            r'\bmaybe\b',
        ]
        for filler in fillers:
            description = re.sub(filler, '', description, flags=re.IGNORECASE)
        description = description.strip()
        if description:
            description = description[0].upper() + description[1:]
        return description
    
    def _extract_assignee(self, sentence: str) -> tuple:
        sentence_lower = sentence.lower()
        for member in self.team_members:
            if member.name.lower() in sentence_lower:
                return member.id, member.name
        return None, None
    
    def _auto_assign(self, description: str) -> tuple:
        description_lower = description.lower()
        skill_keywords = {
            'frontend': ['login', 'ui', 'frontend', 'screen', 'react', 'javascript', 'button', 'form', 'interface'],
            'backend': ['database', 'db', 'query', 'api', 'performance', 'backend', 'server', 'endpoint'],
            'design': ['design', 'onboarding', 'ui/ux', 'figma', 'mockup', 'wireframe', 'prototype'],
            'qa': ['test', 'unit test', 'testing', 'automation', 'qa', 'bug', 'quality'],
        }
        
        best_match = None
        max_score = 0
        match_keywords = []
        
        for member in self.team_members:
            if not member.isActive:
                continue
            
            score = 0
            matched = []
            
            for skill in member.skills:
                skill_lower = skill.lower()
                if skill_lower in description_lower:
                    score += 2
                    matched.append(skill)
            
            member_role_lower = member.role.lower()
            for category, keywords in skill_keywords.items():
                if category in member_role_lower:
                    for keyword in keywords:
                        if keyword in description_lower:
                            score += 1
                            matched.append(keyword)
            
            if score > max_score:
                max_score = score
                best_match = member
                match_keywords = matched
        
        if best_match:
            reasoning = f"Assigned to {best_match.name} based on skills ({', '.join(best_match.skills[:3])}) and keywords ({', '.join(match_keywords[:3])})."
            return best_match.id, best_match.name, reasoning
        
        if self.team_members:
            default = next((m for m in self.team_members if m.isActive), self.team_members[0])
            return default.id, default.name, f"Default assignment to {default.name}."
        
        return None, None, "No team members available."
    
    def _extract_priority(self, sentence: str) -> str:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in self.critical_keywords):
            return "Critical"
        if any(keyword in sentence_lower for keyword in self.high_keywords):
            return "High"
        if any(keyword in sentence_lower for keyword in self.low_keywords):
            return "Low"
        return "Medium"
    
    def _extract_deadline(self, sentence: str) -> tuple:
        sentence_lower = sentence.lower()
        for pattern, days_offset in self.deadline_patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                deadline_text = match.group(0)
                deadline_date = None
                if days_offset is not None:
                    deadline_date = self.meeting_date + timedelta(days=days_offset)
                elif 'friday' in deadline_text:
                    days_ahead = (4 - self.meeting_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    deadline_date = self.meeting_date + timedelta(days=days_ahead)
                elif 'end of' in deadline_text or 'week' in deadline_text:
                    days_ahead = (6 - self.meeting_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    deadline_date = self.meeting_date + timedelta(days=days_ahead)
                elif 'monday' in deadline_text:
                    days_ahead = (7 - self.meeting_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    deadline_date = self.meeting_date + timedelta(days=days_ahead)
                elif 'wednesday' in deadline_text:
                    days_ahead = (2 - self.meeting_date.weekday()) % 7
                    if days_ahead <= 0:
                        days_ahead += 7
                    deadline_date = self.meeting_date + timedelta(days=days_ahead)
                return deadline_text, deadline_date
        return None, None
    
    def _is_blocking(self, sentence: str) -> bool:
        sentence_lower = sentence.lower()
        blocking_keywords = ['blocking', 'blocker', 'blocks users', 'blocks']
        return any(keyword in sentence_lower for keyword in blocking_keywords)
    
    def _extract_dependencies(self, tasks: List[dict]):
        dependency_patterns = [
            r'depends on',
            r'after',
            r'once.*is done',
            r'once.*is completed',
            r'only after',
        ]
        
        for task in tasks:
            sentence_lower = task['originalSentence'].lower()
            has_dependency = any(re.search(pattern, sentence_lower) for pattern in dependency_patterns)
            if has_dependency:
                for other_task in tasks:
                    if other_task == task:
                        continue
                    other_words = set(re.findall(r'\b\w+\b', other_task['description'].lower()))
                    sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
                    common_words = other_words & sentence_words
                    significant_common = [w for w in common_words if len(w) > 3]
                    if len(significant_common) >= 2:
                        if 'id' in other_task:
                            task['dependencies'].append(other_task['id'])

# -----------------------------
# API Routes
# -----------------------------

# Team Members
@api_router.get("/team-members", response_model=List[TeamMember])
async def get_team_members():
    members = await db.team_members.find({}, {"_id": 0}).to_list(1000)
    for member in members:
        if isinstance(member.get('createdAt'), str):
            member['createdAt'] = datetime.fromisoformat(member['createdAt'])
        if isinstance(member.get('updatedAt'), str):
            member['updatedAt'] = datetime.fromisoformat(member['updatedAt'])
    return members

@api_router.post("/team-members", response_model=TeamMember)
async def create_team_member(input: TeamMemberCreate):
    member = TeamMember(**input.model_dump())
    doc = member.model_dump()
    doc['createdAt'] = doc['createdAt'].isoformat()
    doc['updatedAt'] = doc['updatedAt'].isoformat()
    await db.team_members.insert_one(doc)
    return member

@api_router.put("/team-members/{member_id}", response_model=TeamMember)
async def update_team_member(member_id: str, input: TeamMemberUpdate):
    existing = await db.team_members.find_one({"id": member_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Team member not found")
    
    update_data = {k: v for k, v in input.model_dump().items() if v is not None}
    update_data['updatedAt'] = datetime.now(timezone.utc).isoformat()
    
    await db.team_members.update_one({"id": member_id}, {"$set": update_data})
    
    updated = await db.team_members.find_one({"id": member_id}, {"_id": 0})
    if isinstance(updated.get('createdAt'), str):
        updated['createdAt'] = datetime.fromisoformat(updated['createdAt'])
    if isinstance(updated.get('updatedAt'), str):
        updated['updatedAt'] = datetime.fromisoformat(updated['updatedAt'])
    return TeamMember(**updated)

@api_router.patch("/team-members/{member_id}/deactivate")
async def deactivate_team_member(member_id: str):
    result = await db.team_members.update_one(
        {"id": member_id},
        {"$set": {"isActive": False, "updatedAt": datetime.now(timezone.utc).isoformat()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Team member not found")
    return {"message": "Team member deactivated"}

# Meetings
@api_router.post("/meetings", response_model=Meeting)
async def create_meeting(
    title: str = Form(...),
    date: Optional[str] = Form(None),
    audio: UploadFile = File(...)
):
    meeting_date = datetime.now(timezone.utc)
    if date:
        try:
            meeting_date = datetime.fromisoformat(date)
        except:
            pass
    
    meeting_id = str(uuid.uuid4())
    file_extension = Path(audio.filename).suffix or ".wav"
    audio_filename = f"{meeting_id}{file_extension}"
    audio_path = UPLOADS_DIR / audio_filename
    
    # Save uploaded file
    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)
    
    meeting = Meeting(
        id=meeting_id,
        title=title,
        date=meeting_date,
        audioUrl=str(audio_path),
        status="uploaded"
    )
    
    doc = meeting.model_dump()
    doc['date'] = doc['date'].isoformat()
    doc['createdAt'] = doc['createdAt'].isoformat()
    doc['updatedAt'] = doc['updatedAt'].isoformat()
    
    await db.meetings.insert_one(doc)
    return meeting

# Helper: synchronous transcription done inside thread executor to avoid blocking
def _transcribe_file_sync(path: str):
    with open(path, "rb") as audio_file:
        # openai.Audio.transcribe returns a dict-like or object with 'text'
        return openai.Audio.transcribe(model="whisper-1", file=audio_file)

@api_router.post("/meetings/{meeting_id}/transcribe", response_model=Meeting)
async def transcribe_meeting(meeting_id: str):
    meeting = await db.meetings.find_one({"id": meeting_id}, {"_id": 0})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    await db.meetings.update_one(
        {"id": meeting_id},
        {"$set": {"status": "transcribing", "updatedAt": datetime.now(timezone.utc).isoformat()}}
    )
    
    try:
        audio_path = meeting['audioUrl']
        # Run transcription in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, _transcribe_file_sync, audio_path)
        
        # response might be dict-like or object; try to get text robustly
        transcript = None
        if isinstance(response, dict):
            transcript = response.get("text") or response.get("transcript") or str(response)
        else:
            # some SDKs return object with .text
            transcript = getattr(response, "text", None) or str(response)
        
        # store transcript and update meeting
        await db.meetings.update_one(
            {"id": meeting_id},
            {"$set": {
                "transcript": transcript,
                "status": "transcribed",
                "updatedAt": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        updated = await db.meetings.find_one({"id": meeting_id}, {"_id": 0})
        if isinstance(updated.get('date'), str):
            updated['date'] = datetime.fromisoformat(updated['date'])
        if isinstance(updated.get('createdAt'), str):
            updated['createdAt'] = datetime.fromisoformat(updated['createdAt'])
        if isinstance(updated.get('updatedAt'), str):
            updated['updatedAt'] = datetime.fromisoformat(updated['updatedAt'])
        
        return Meeting(**updated)
        
    except Exception as e:
        await db.meetings.update_one(
            {"id": meeting_id},
            {"$set": {"status": "error", "updatedAt": datetime.now(timezone.utc).isoformat()}}
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@api_router.post("/meetings/{meeting_id}/process-tasks", response_model=List[Task])
async def process_meeting_tasks(meeting_id: str):
    meeting = await db.meetings.find_one({"id": meeting_id}, {"_id": 0})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    if not meeting.get('transcript'):
        raise HTTPException(status_code=400, detail="Meeting has no transcript. Please transcribe first.")
    
    await db.meetings.update_one(
        {"id": meeting_id},
        {"$set": {"status": "processing", "updatedAt": datetime.now(timezone.utc).isoformat()}}
    )
    
    try:
        team_members_data = await db.team_members.find({}, {"_id": 0}).to_list(1000)
        team_members = []
        for tm_data in team_members_data:
            if isinstance(tm_data.get('createdAt'), str):
                tm_data['createdAt'] = datetime.fromisoformat(tm_data['createdAt'])
            if isinstance(tm_data.get('updatedAt'), str):
                tm_data['updatedAt'] = datetime.fromisoformat(tm_data['updatedAt'])
            team_members.append(TeamMember(**tm_data))
        
        meeting_date = meeting['date']
        if isinstance(meeting_date, str):
            meeting_date = datetime.fromisoformat(meeting_date)
        
        extractor = TaskExtractor(team_members, meeting_date)
        extracted_tasks = extractor.extract_tasks(meeting['transcript'])
        
        await db.tasks.delete_many({"meetingId": meeting_id})
        
        tasks = []
        for task_data in extracted_tasks:
            task_data.pop('originalSentence', None)
            # convert deadlineDate back to datetime if iso string
            if task_data.get('deadlineDate') and isinstance(task_data['deadlineDate'], str):
                try:
                    task_data['deadlineDate'] = datetime.fromisoformat(task_data['deadlineDate'])
                except:
                    task_data['deadlineDate'] = None
            
            task = Task(meetingId=meeting_id, **task_data)
            
            doc = task.model_dump()
            doc['createdAt'] = doc['createdAt'].isoformat()
            doc['updatedAt'] = doc['updatedAt'].isoformat()
            if doc.get('deadlineDate') and isinstance(doc['deadlineDate'], datetime):
                doc['deadlineDate'] = doc['deadlineDate'].isoformat()
            
            await db.tasks.insert_one(doc)
            tasks.append(task)
        
        await db.meetings.update_one(
            {"id": meeting_id},
            {"$set": {"status": "completed", "updatedAt": datetime.now(timezone.utc).isoformat()}}
        )
        
        return tasks
        
    except Exception as e:
        await db.meetings.update_one(
            {"id": meeting_id},
            {"$set": {"status": "error", "updatedAt": datetime.now(timezone.utc).isoformat()}}
        )
        raise HTTPException(status_code=500, detail=f"Task processing failed: {str(e)}")

@api_router.get("/meetings", response_model=List[MeetingSummary])
async def get_meetings():
    pipeline = [
        {
            "$lookup": {
                "from": "tasks",
                "localField": "id",
                "foreignField": "meetingId",
                "as": "tasks"
            }
        },
        {
            "$addFields": {
                "taskCount": {"$size": "$tasks"}
            }
        },
        {
            "$project": {
                "tasks": 0,
                "_id": 0
            }
        }
    ]
    
    meetings = await db.meetings.aggregate(pipeline).to_list(1000)
    
    summaries = []
    for meeting in meetings:
        if isinstance(meeting.get('date'), str):
            meeting['date'] = datetime.fromisoformat(meeting['date'])
        
        summaries.append(MeetingSummary(
            id=meeting['id'],
            title=meeting['title'],
            date=meeting['date'],
            status=meeting['status'],
            taskCount=meeting.get('taskCount', 0)
        ))
    
    return summaries

@api_router.get("/meetings/{meeting_id}", response_model=Meeting)
async def get_meeting(meeting_id: str):
    meeting = await db.meetings.find_one({"id": meeting_id}, {"_id": 0})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    if isinstance(meeting.get('date'), str):
        meeting['date'] = datetime.fromisoformat(meeting['date'])
    if isinstance(meeting.get('createdAt'), str):
        meeting['createdAt'] = datetime.fromisoformat(meeting['createdAt'])
    if isinstance(meeting.get('updatedAt'), str):
        meeting['updatedAt'] = datetime.fromisoformat(meeting['updatedAt'])
    
    return Meeting(**meeting)

# Tasks
@api_router.get("/meetings/{meeting_id}/tasks", response_model=List[Task])
async def get_meeting_tasks(meeting_id: str):
    tasks = await db.tasks.find({"meetingId": meeting_id}, {"_id": 0}).to_list(1000)
    
    for task in tasks:
        if isinstance(task.get('createdAt'), str):
            task['createdAt'] = datetime.fromisoformat(task['createdAt'])
        if isinstance(task.get('updatedAt'), str):
            task['updatedAt'] = datetime.fromisoformat(task['updatedAt'])
        if isinstance(task.get('deadlineDate'), str):
            task['deadlineDate'] = datetime.fromisoformat(task['deadlineDate'])
    
    return tasks

@api_router.patch("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, input: TaskUpdate):
    existing = await db.tasks.find_one({"id": task_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Task not found")
    
    update_data = {k: v for k, v in input.model_dump().items() if v is not None}
    update_data['updatedAt'] = datetime.now(timezone.utc).isoformat()
    
    if 'deadlineDate' in update_data and isinstance(update_data['deadlineDate'], datetime):
        update_data['deadlineDate'] = update_data['deadlineDate'].isoformat()
    
    if 'assignedToMemberId' in update_data:
        member = await db.team_members.find_one({"id": update_data['assignedToMemberId']}, {"_id": 0})
        if member:
            update_data['assignedToName'] = member['name']
    
    await db.tasks.update_one({"id": task_id}, {"$set": update_data})
    
    updated = await db.tasks.find_one({"id": task_id}, {"_id": 0})
    if isinstance(updated.get('createdAt'), str):
        updated['createdAt'] = datetime.fromisoformat(updated['createdAt'])
    if isinstance(updated.get('updatedAt'), str):
        updated['updatedAt'] = datetime.fromisoformat(updated['updatedAt'])
    if isinstance(updated.get('deadlineDate'), str):
        updated['deadlineDate'] = datetime.fromisoformat(updated['deadlineDate'])
    
    return Task(**updated)

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
