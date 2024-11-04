"""
Django Integration Example
=========================
Example of using AI Orchestrator with Django.
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from ai_orchestrator import AIOrchestrator
import asyncio

orchestrator = AIOrchestrator(api_key="your-api-key")

@csrf_exempt
def process_text(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            result = asyncio.run(orchestrator.process_input(
                session_id=data.get('session_id'),
                user_input=data.get('text'),
                context_updates=data.get('context')
            ))
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)