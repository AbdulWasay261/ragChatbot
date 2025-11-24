from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatSession, Message
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  
import json
import uuid




model = OllamaLLM(model="llama3.2")

template = """
You are an expert Coursera advisor.
keep in mind if chat history
Chat History:
{chat_history}


Your answer must ONLY use the course data below:
{db_text}

Here is the new question:
{question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def chat_view(request):
    # Get session_id from Django session
    session_id = request.session.get('chat_session_id')

    if not session_id:
        # Generate a new session ID if none exists
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id

    # Get existing ChatSession or create a new one
    chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)

    messages = chat_session.messages.all()

    return render(request, 'chat/chat.html', {
        'chat_session': chat_session,
        'messages': messages
    })
@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('message', '')
        session_id = request.session.get('chat_session_id')
        
        if not session_id:
            return JsonResponse({'error': 'No session found'}, status=400)
        
        chat_session = ChatSession.objects.get(session_id=session_id)
        
        # Save user message
        Message.objects.create(
            session=chat_session,
            role='user',
            content=question
        )
        
        # Get chat history
        messages = list(chat_session.messages.all())
        chat_history = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[:-1]])
        
        # Retrieve relevant documents
        
        
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"Doc {i}: {doc.metadata.get('name', 'Unknown')}")
        
        db_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        result = chain.invoke({
            "question": question,
            "db_text": db_text,
            "chat_history": chat_history
        })
        
        # Save assistant message
        Message.objects.create(
            session=chat_session,
            role='assistant',
            content=result
        )
        
        return JsonResponse({
            'response': result,
            'success': True
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
@csrf_exempt
def clear_chat(request):
    session_id = request.session.get('chat_session_id')
    if session_id:
        chat_session = ChatSession.objects.get(session_id=session_id)
        chat_session.messages.all().delete()
    return JsonResponse({'success': True})