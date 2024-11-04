"""
Flask Integration Example
========================
Example of using AI Orchestrator with Flask.
"""

from flask import Flask, request, jsonify
from ai_orchestrator import AIOrchestrator
import asyncio
from functools import wraps

app = Flask(__name__)
orchestrator = AIOrchestrator(api_key="your-api-key")

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

@app.route('/analyze', methods=['POST'])
@async_route
async def analyze():
    data = request.get_json()
    try:
        result = await orchestrator.process_input(
            session_id=data.get('session_id'),
            user_input=data.get('text'),
            context_updates=data.get('context')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)