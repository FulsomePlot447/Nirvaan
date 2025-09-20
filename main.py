from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import os
import tempfile
import PyPDF2
import docx
import pandas as pd
from typing import List, Optional
import json
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from datetime import datetime
import io

load_dotenv()

# -------------------------
# PDF Generation Model
# -------------------------
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class PDFRequest(BaseModel):
    messages: List[Message]

def parse_table_content(content: str) -> List[List[str]]:
    """Parse table content from text with | separators"""
    lines = content.split('\n')
    table_data = []
    
    for line in lines:
        line = line.strip()
        # Skip separator lines (containing only +, -, |, spaces)
        if line and not all(c in '+- |' for c in line):
            # Check if this looks like a table row
            if '|' in line:
                # Split by | and clean up
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at start/end (from leading/trailing |)
                cells = [cell for cell in cells if cell]
                if cells:
                    table_data.append(cells)
    
    return table_data

def detect_table_in_content(content: str) -> bool:
    """Detect if content contains a table structure"""
    lines = content.split('\n')
    pipe_lines = 0
    separator_lines = 0
    
    for line in lines:
        if '|' in line and len([c for c in line if c == '|']) >= 2:
            pipe_lines += 1
        elif line.strip() and all(c in '+- |' for c in line.strip()):
            separator_lines += 1
    
    # Consider it a table if we have multiple pipe lines and at least one separator
    return pipe_lines >= 2 and separator_lines >= 1

def generate_pdf(messages: List[Message]) -> bytes:
    """Generate PDF from chat messages with proper table formatting"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=HexColor('#1f2937'),
        spaceAfter=20,
    )
    
    user_style = ParagraphStyle(
        'UserMessage',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#2563eb'),
        backgroundColor=HexColor('#eff6ff'),
        borderPadding=10,
        borderWidth=1,
        borderColor=HexColor('#93c5fd'),
        spaceAfter=10,
        leftIndent=20,
    )
    
    ai_style = ParagraphStyle(
        'AIMessage',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#374151'),
        backgroundColor=HexColor('#f9fafb'),
        borderPadding=10,
        borderWidth=1,
        borderColor=HexColor('#d1d5db'),
        spaceAfter=10,
        rightIndent=20,
    )
    
    system_style = ParagraphStyle(
        'SystemMessage',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#6b7280'),
        backgroundColor=HexColor('#f3f4f6'),
        borderPadding=8,
        borderWidth=1,
        borderColor=HexColor('#e5e7eb'),
        spaceAfter=10,
        alignment=1,  # Center alignment
    )
    
    # Table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e5e7eb')),  # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1f2937')),   # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),                 # Center alignment
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),       # Header font
        ('FONTSIZE', (0, 0), (-1, 0), 10),                     # Header font size
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),           # Body font
        ('FONTSIZE', (0, 1), (-1, -1), 9),                     # Body font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),                # Header padding
        ('TOPPADDING', (0, 0), (-1, 0), 12),                   # Header padding
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),                # Body padding
        ('TOPPADDING', (0, 1), (-1, -1), 8),                   # Body padding
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#9ca3af')),    # Grid lines
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),                # Vertical alignment
    ])
    
    # Build PDF content
    story = []
    
    # Title
    story.append(Paragraph("NIRVAAN AI ASSISTANT - Chat Export", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Messages
    for i, msg in enumerate(messages):
        if msg.role == "user":
            story.append(Paragraph(f"<b>You:</b> {msg.content}", user_style))
        elif msg.role == "bot":
            # Check if the bot message contains a table
            if detect_table_in_content(msg.content):
                # Add the AI label
                story.append(Paragraph("<b>AI Assistant:</b>", ai_style))
                
                # Split content by potential table sections
                parts = msg.content.split('\n\n')
                
                for part in parts:
                    if detect_table_in_content(part):
                        # Parse and create table
                        table_data = parse_table_content(part)
                        if table_data:
                            # Create ReportLab Table
                            pdf_table = Table(table_data)
                            pdf_table.setStyle(table_style)
                            story.append(pdf_table)
                            story.append(Spacer(1, 0.2*inch))
                    else:
                        # Regular text content
                        if part.strip():
                            story.append(Paragraph(part, ai_style))
            else:
                # Regular AI response without table
                story.append(Paragraph(f"<b>AI Assistant:</b> {msg.content}", ai_style))
        elif msg.role == "system":
            story.append(Paragraph(f"<i>System:</i> {msg.content}", system_style))
        
        if msg.timestamp:
            story.append(Paragraph(f"<font size='8' color='#9ca3af'>{msg.timestamp}</font>", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# -------------------------
# File Processing Functions
# -------------------------
def process_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def process_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error processing DOCX: {str(e)}"


def process_csv(file_path: str) -> str:
    """Extract data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        preview = df.head(10).to_string()
        info = f"Shape: {df.shape}, Columns: {list(df.columns)}"
        return f"CSV Info: {info}\n\nPreview:\n{preview}"
    except Exception as e:
        return f"Error processing CSV: {str(e)}"


def process_txt(file_path: str) -> str:
    """Read text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error processing TXT: {str(e)}"


def process_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """Process uploaded file and return content and file type"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_file.write(file.file.read())
        temp_path = temp_file.name

    try:
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            content = process_pdf(temp_path)
        elif file_extension == 'docx':
            content = process_docx(temp_path)
        elif file_extension == 'csv':
            content = process_csv(temp_path)
        elif file_extension in ['txt', 'md']:
            content = process_txt(temp_path)
        else:
            content = "Unsupported file type"
        
        return content, file_extension
    finally:
        os.unlink(temp_path)

# -------------------------
# Timetable Formatting Functions
# -------------------------
def detect_timetable_request(query: str) -> bool:
    """Detect if the user is asking for a timetable"""
    timetable_keywords = [
        'timetable', 'time table', 'schedule', 'planner', 'calendar',
        'weekly schedule', 'daily schedule', 'class schedule', 'work schedule',
        'routine', 'agenda', 'plan', 'timeline', 'itinerary'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in timetable_keywords)

def format_timetable_response(response_text: str) -> str:
    """Enhance timetable responses with better formatting"""
    # If response contains table-like structure, enhance it
    lines = response_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Add borders to table-like structures
        if '|' in line and len([x for x in line if x == '|']) >= 2:
            # This looks like a table row
            if not line.strip().startswith('|'):
                line = '| ' + line + ' |'
            formatted_lines.append(line)
        elif line.strip() and all(c in '-+=| ' for c in line.strip()):
            # This looks like a separator line
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def create_timetable_prompt(query: str) -> str:
    """Create an enhanced prompt for timetable generation"""
    return f"""You are an AI assistant specialized in creating well-formatted timetables and schedules. 

When creating timetables, always format them as proper tables with:
1. Clear headers with column names
2. Proper alignment using | symbols
3. Separator lines using dashes (-)
4. Consistent spacing

Example format:
```
+----------+----------+----------+----------+----------+
| Time     | Monday   | Tuesday  | Wednesday| Thursday |
+----------+----------+----------+----------+----------+
| 9:00 AM  | Math     | English  | Science  | Math     |
| 10:00 AM | History  | Math     | English  | Art      |
| 11:00 AM | Science  | Art      | History  | English  |
+----------+----------+----------+----------+----------+
```

Or use simple markdown table format:
```
| Time     | Monday   | Tuesday  | Wednesday | Thursday |
|----------|----------|----------|-----------|----------|
| 9:00 AM  | Math     | English  | Science   | Math     |
| 10:00 AM | History  | Math     | English   | Art      |
| 11:00 AM | Science  | Art      | History   | English  |
```

User's request: {query}

Please create a well-formatted timetable based on this request. Make sure to use proper table formatting with borders and clear structure."""

# -------------------------
# LLM Setup with Enhanced Timetable Support
# -------------------------
def create_enhanced_llm():
    """Create LLM with system message for better timetable formatting"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        print("✅ OpenAI LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing OpenAI LLM: {e}")
        print("Note: Make sure you have OPENAI_API_KEY in your .env file")
        # Create a mock LLM for testing
        class MockLLM:
            def stream(self, messages):
                query_text = ""
                for msg in messages:
                    if hasattr(msg, 'content'):
                        query_text += msg.content
                
                if detect_timetable_request(query_text):
                    mock_response = """Here's a sample timetable:

+----------+----------+----------+----------+----------+
| Time     | Monday   | Tuesday  | Wednesday| Thursday |
+----------+----------+----------+----------+----------+
| 9:00 AM  | Math     | English  | Science  | Math     |
| 10:00 AM | History  | Math     | English  | Art      |
| 11:00 AM | Science  | Art      | History  | English  |
+----------+----------+----------+----------+----------+

This is a mock timetable response for testing purposes."""
                else:
                    mock_response = f"Mock response to: {query_text}"
                
                words = mock_response.split()
                for word in words:
                    yield type('obj', (object,), {'content': word + ' '})
        
        llm = MockLLM()
        print("🔄 Using mock LLM for testing")
        return llm

llm = create_enhanced_llm()

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI(title="NIRVAAN AI ASSISTANT", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation contexts
conversation_contexts = {}

# -------------------------
# Context Management Functions
# -------------------------
def detect_topic_change(current_query: str, conversation_history: List[dict]) -> bool:
    """Detect if the user is changing topics"""
    if not conversation_history:
        return False
    
    # Keywords that indicate topic change
    topic_change_indicators = [
        "let's talk about", "change topic", "new topic", "different question",
        "switch to", "move on to", "now about", "forget about", "ignore previous",
        "start fresh", "new conversation", "different subject", "something else"
    ]
    
    query_lower = current_query.lower()
    
    # Check for explicit topic change indicators
    for indicator in topic_change_indicators:
        if indicator in query_lower:
            return True
    
    # If query is very different from recent context (simple heuristic)
    if len(conversation_history) > 0:
        last_messages = conversation_history[-3:]  # Last 3 messages
        context_text = " ".join([msg.get("content", "") for msg in last_messages]).lower()
        
        # Simple word overlap check
        query_words = set(query_lower.split())
        context_words = set(context_text.split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "will", "would", "could", "should", "can", "may", "might", "what", "how", "when", "where", "why", "who"}
        query_words = query_words - common_words
        context_words = context_words - common_words
        
        if len(query_words) > 2 and len(context_words) > 2:
            overlap = len(query_words.intersection(context_words))
            if overlap / len(query_words) < 0.2:  # Less than 20% word overlap
                return True
    
    return False

def get_conversation_context(session_id: str) -> str:
    """Get formatted conversation context for the session"""
    if session_id not in conversation_contexts:
        return ""
    
    history = conversation_contexts[session_id]
    if not history:
        return ""
    
    # Format recent conversation history
    context = "\n\nConversation Context:\n"
    for msg in history[-6:]:  # Last 6 messages for context
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            context += f"User: {content}\n"
        elif role == "assistant":
            context += f"Assistant: {content[:200]}{'...' if len(content) > 200 else ''}\n"
    
    context += "\nPlease continue this conversation and stay on the same topic unless the user explicitly asks to change topics or asks something completely unrelated.\n"
    return context

def add_to_conversation_history(session_id: str, role: str, content: str):
    """Add message to conversation history"""
    if session_id not in conversation_contexts:
        conversation_contexts[session_id] = []
    
    conversation_contexts[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 20 messages to prevent memory issues
    if len(conversation_contexts[session_id]) > 20:
        conversation_contexts[session_id] = conversation_contexts[session_id][-20:]

def clear_conversation_context(session_id: str):
    """Clear conversation context for a session"""
    if session_id in conversation_contexts:
        conversation_contexts[session_id] = []

# Store processed files content temporarily
processed_files = {}

# -------------------------
# Enhanced Models
# -------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    file_context: Optional[str] = None

# -------------------------
# Health Check Endpoint
# -------------------------
@app.get("/")
async def health_check():
    return {"status": "running", "message": "NIRVAAN AI ASSISTANT API is running!"}

# -------------------------
# File Upload Endpoint
# -------------------------
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    results = []
    
    for file in files:
        try:
            # Check file size (limit to 10MB)
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)
            
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "File too large (max 10MB)"
                })
                continue
            
            content, file_type = process_uploaded_file(file)
            
            # Store processed content
            file_id = f"{file.filename}_{len(processed_files)}"
            processed_files[file_id] = {
                "filename": file.filename,
                "content": content,
                "file_type": file_type
            }
            
            results.append({
                "filename": file.filename,
                "file_id": file_id,
                "file_type": file_type,
                "content_preview": content[:500] + "..." if len(content) > 500 else content,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {"results": results, "total_files": len(processed_files)}

# -------------------------
# Enhanced Stream Chat Endpoint with Timetable Support
# -------------------------
@app.post("/stream")
async def stream_chat(query: str = Form(...), session_id: str = Form("default"), file_context: Optional[str] = Form(None)):
    """Stream chat response with enhanced timetable formatting"""
    def generate():
        try:
            # Check for topic change
            current_history = conversation_contexts.get(session_id, [])
            topic_changed = detect_topic_change(query, current_history)
            
            if topic_changed:
                # Clear context if topic changed
                clear_conversation_context(session_id)
                current_history = []
            
            # Add user message to conversation history
            add_to_conversation_history(session_id, "user", query)
            
            # Get conversation context
            context = get_conversation_context(session_id)
            
            # Prepare file context from uploaded files
            file_info = ""
            if file_context and processed_files:
                file_info = "\n\nUploaded Files Context:\n"
                for file_id, file_data in processed_files.items():
                    file_info += f"\n--- {file_data['filename']} ({file_data['file_type']}) ---\n"
                    file_info += file_data['content'][:2000] + ("..." if len(file_data['content']) > 2000 else "")
                    file_info += "\n"
            
            # Create messages for the LLM
            messages = []
            
            # System message for timetable formatting
            system_message = SystemMessage(content="""You are NIRVAAN AI Assistant, an intelligent assistant that excels at creating well-formatted timetables and schedules.

When users request timetables, schedules, or planners, always format them using proper table structure with:
- Clear headers
- Consistent column alignment using | symbols
- Separator lines using + and - symbols
- Proper spacing for readability

Use this format for timetables:
```
+----------+----------+----------+----------+
| Time     | Monday   | Tuesday  | Wednesday|
+----------+----------+----------+----------+
| 9:00 AM  | Subject1 | Subject2 | Subject3 |
| 10:00 AM | Subject4 | Subject5 | Subject6 |
+----------+----------+----------+----------+
```

For all responses, be helpful, accurate, and maintain conversation context appropriately.""")
            
            messages.append(system_message)
            
            # Check if this is a timetable request
            if detect_timetable_request(query):
                enhanced_query = create_timetable_prompt(query)
            else:
                enhanced_query = f"{query}{context}{file_info}"
            
            # Human message
            messages.append(HumanMessage(content=enhanced_query))
            
            # Generate response
            full_response = ""
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
        # Replace markdown bullets with actual line breaks
                    text = chunk.content.replace("\n", "\n")
                    full_response += text
                    yield text
            
            # Format timetable if needed
            if detect_timetable_request(query):
                full_response = format_timetable_response(full_response)
            
            # Add assistant response to conversation history
            add_to_conversation_history(session_id, "assistant", full_response)
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            add_to_conversation_history(session_id, "assistant", error_msg)
            yield error_msg
    
    return StreamingResponse(generate(), media_type="text/plain")

# -------------------------
# File Management Endpoints
# -------------------------
@app.get("/files")
async def list_files():
    """List all uploaded files"""
    return {
        "files": [
            {
                "file_id": file_id,
                "filename": data["filename"],
                "file_type": data["file_type"],
                "content_length": len(data["content"])
            }
            for file_id, data in processed_files.items()
        ]
    }

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a specific uploaded file"""
    if file_id in processed_files:
        del processed_files[file_id]
        return {"message": f"File {file_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.delete("/files")
async def clear_all_files():
    """Clear all uploaded files"""
    processed_files.clear()
    return {"message": "All files cleared successfully"}

# -------------------------
# Context Management Endpoints
# -------------------------
@app.post("/clear-context")
async def clear_context(session_id: str = Form("default")):
    """Clear conversation context for a session"""
    clear_conversation_context(session_id)
    return {"message": f"Conversation context cleared for session {session_id}"}

@app.get("/context/{session_id}")
async def get_context(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "history": conversation_contexts.get(session_id, [])
    }

# -------------------------
# PDF Generation Endpoint
# -------------------------
@app.post("/generate-pdf")
async def generate_conversation_pdf(request: PDFRequest):
    """Generate PDF from chat conversation"""
    try:
        pdf_bytes = generate_pdf(request.messages)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=chat_conversation.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NIRVAAN AI ASSISTANT server...")
    print("📝 Make sure you have a .env file with OPENAI_API_KEY")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("📄 PDF generation endpoint: /generate-pdf")
    print("📅 Enhanced with timetable formatting support!")
    uvicorn.run(app, host="0.0.0.0", port=8000)