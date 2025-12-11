"""
Manual Intervention Labeler - A Flask web application for manually selecting
preferred interventions from training_generations_hidden_prompt_gpt_4-1-mini_3x.jsonl
"""

import json
import os
import random
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
from src.utils import load_dataset_from_jsonl
from src.agent import AGENT_INTERVENTION_PROMPT_TEMPLATE, AGENT_OUTPUT_JSON
from src.utils import format_thread_for_prompt

app = Flask(__name__)

# Configuration
INTERVENTIONS_FILE = 'results/training_candidates.jsonl'
DOCUMENTS_FILE = 'results/training_docs.jsonl'

# Derive output and progress files from interventions file
_base_name = os.path.splitext(INTERVENTIONS_FILE)[0]
OUTPUT_FILE = f'{_base_name}_manual_labels.jsonl'
PROGRESS_FILE = f'{_base_name}_manual_labels_progress.json'

# Global state
intervention_pairs = []
documents = []
progress = {'current_index': 0, 'labeled': []}


def load_data():
    """Load intervention pairs and documents."""
    global intervention_pairs, documents, progress

    # Load intervention pairs
    with open(INTERVENTIONS_FILE, 'r', encoding='utf-8') as f:
        intervention_pairs = [json.loads(line) for line in f if line.strip()]

    # Load documents
    documents = load_dataset_from_jsonl(DOCUMENTS_FILE)

    # Load progress if exists
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    else:
        progress = {'current_index': 0, 'labeled': []}

    print(f"Loaded {len(intervention_pairs)} intervention pairs")
    print(f"Loaded {len(documents)} documents")
    print(f"Progress: {len(progress['labeled'])} labeled, current index: {progress['current_index']}")


def save_progress():
    """Save current progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def save_label(result):
    """Append a labeled result to the output file."""
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def get_prompt_for_pair(pair, entry):
    """Generate the prompt that would be shown to the agent."""
    n_comments = pair['comments_used']
    partial_thread = entry.comment_thread[:n_comments]

    prompt = AGENT_INTERVENTION_PROMPT_TEMPLATE.format(
        document=entry.document,
        highlighted_sentence=entry.highlighted_sentence,
        thread=format_thread_for_prompt(partial_thread),
        intervention_instruction="",
        output_format=AGENT_OUTPUT_JSON
    )
    return prompt


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Intervention Labeler</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .progress-bar {
            background: rgba(255,255,255,0.3);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }
        .progress-fill {
            background: rgba(255,255,255,0.9);
            height: 100%;
            transition: width 0.3s ease;
        }
        .progress-text {
            margin-top: 5px;
            font-size: 14px;
        }
        .context-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .context-section h2 {
            color: #333;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .document {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
        }
        .highlighted {
            background: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }
        .thread {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
        }
        .thread-comment {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        .thread-comment.peer {
            border-left: 3px solid #17a2b8;
        }
        .thread-comment.author {
            border-left: 3px solid #28a745;
        }
        .speaker {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .interventions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .intervention-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
            border: 3px solid transparent;
        }
        .intervention-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .intervention-card.selected {
            border-color: #28a745;
            background: #f0fff4;
        }
        .intervention-card h3 {
            margin-top: 0;
            color: #333;
        }
        .intervention-type {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .type-compromise {
            background: #d4edda;
            color: #155724;
        }
        .type-socratic {
            background: #cce5ff;
            color: #004085;
        }
        .type-no_intervention {
            background: #f8d7da;
            color: #721c24;
        }
        .intervention-text {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            font-size: 14px;
            max-height: 250px;
            overflow-y: auto;
        }
        .actions {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5a6fd6;
        }
        .btn-primary:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .btn-skip {
            background: #ffc107;
            color: #212529;
        }
        .btn-skip:hover {
            background: #e0a800;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .metadata {
            color: #666;
            font-size: 14px;
        }
        .complete-message {
            text-align: center;
            padding: 50px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .complete-message h2 {
            color: #28a745;
        }
        .keyboard-hint {
            background: #e8f4f8;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
        }
        kbd {
            background: #333;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Manual Intervention Labeler</h1>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ progress_pct }}%"></div>
        </div>
        <div class="progress-text">
            {{ labeled_count }} / {{ total_count }} labeled ({{ progress_pct|round(1) }}%) |
            Viewing: {{ current_index + 1 }} / {{ total_count }}
        </div>
    </div>

    {% if current_index >= total_count %}
    <div class="complete-message">
        <h2>🎉 All Done!</h2>
        <p>You have labeled all {{ total_count }} intervention pairs.</p>
        <p>Results saved to: <code>{{ output_file }}</code></p>
        <div class="actions">
            <a href="{{ url_for('goto', index=0) }}" class="btn btn-secondary">Review from Start</a>
        </div>
    </div>
    {% else %}

    <div class="keyboard-hint">
        Keyboard shortcuts: <kbd>1</kbd> Select Option A (Left) | <kbd>2</kbd> Select Option B (Right) |
        <kbd>S</kbd> Skip (Equal) | <kbd>Enter</kbd> Submit | <kbd>←</kbd> Previous | <kbd>→</kbd> Next
    </div>

    <div class="nav-buttons">
        {% if current_index > 0 %}
        <a href="{{ url_for('goto', index=current_index - 1) }}" class="btn btn-secondary">← Previous</a>
        {% endif %}
        {% if current_index < total_count - 1 %}
        <a href="{{ url_for('goto', index=current_index + 1) }}" class="btn btn-secondary">Next →</a>
        {% endif %}
        <a href="{{ url_for('goto', index=next_unlabeled) }}" class="btn btn-primary">Go to Next Unlabeled</a>
    </div>

    <div class="context-section">
        <h2>📄 Document Context</h2>
        <div class="metadata">
            Doc Index: {{ pair.doc_index }} | Scenario: {{ pair.scenario }} | Comments Used: {{ pair.comments_used }}
            {% if is_labeled %} | <span style="color: #28a745; font-weight: bold;">✓ Already Labeled</span>{% endif %}
        </div>
        <div class="document">{{ document }}</div>
    </div>

    <div class="context-section">
        <h2>🔍 Highlighted Sentence</h2>
        <div class="highlighted">{{ highlighted_sentence }}</div>
    </div>

    <div class="context-section">
        <h2>💬 Comment Thread ({{ pair.comments_used }} comments)</h2>
        <div class="thread">
            {% for comment in thread %}
            <div class="thread-comment {{ comment.speaker }}">
                <div class="speaker">{{ comment.speaker|title }}:</div>
                <div>{{ comment.text }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

    <h2>Choose the Better Intervention</h2>

    <form id="labelForm" method="POST" action="{{ url_for('label') }}">
        <input type="hidden" name="index" value="{{ current_index }}">
        <input type="hidden" name="choice" id="choiceInput" value="">

        <div class="interventions">
            <div class="intervention-card" id="cardLeft" onclick="selectIntervention({{ left_idx }})">
                <h3>Option A</h3>
                <span class="intervention-type type-{{ left_type }}">{{ left_type }}</span>
                <div class="intervention-text">{{ left_text }}</div>
            </div>

            <div class="intervention-card" id="cardRight" onclick="selectIntervention({{ right_idx }})">
                <h3>Option B</h3>
                <span class="intervention-type type-{{ right_type }}">{{ right_type }}</span>
                <div class="intervention-text">{{ right_text }}</div>
            </div>
        </div>

        <div class="actions">
            <button type="submit" class="btn btn-primary" id="submitBtn" disabled>Submit Selection</button>
            <button type="button" class="btn btn-skip" onclick="skipEqual()">Skip (Equal Quality)</button>
        </div>
    </form>

    <script>
        let selectedChoice = null;
        const leftIdx = {{ left_idx }};
        const rightIdx = {{ right_idx }};

        function selectIntervention(choice) {
            selectedChoice = choice;
            document.getElementById('choiceInput').value = choice;
            document.getElementById('cardLeft').classList.toggle('selected', choice === leftIdx);
            document.getElementById('cardRight').classList.toggle('selected', choice === rightIdx);
            document.getElementById('submitBtn').disabled = false;
        }

        function skipEqual() {
            document.getElementById('choiceInput').value = '0';
            document.getElementById('labelForm').submit();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === '1') {
                selectIntervention(leftIdx);  // Select left option
            } else if (e.key === '2') {
                selectIntervention(rightIdx);  // Select right option
            } else if (e.key === 's' || e.key === 'S') {
                skipEqual();
            } else if (e.key === 'Enter' && selectedChoice) {
                document.getElementById('labelForm').submit();
            } else if (e.key === 'ArrowLeft') {
                {% if current_index > 0 %}
                window.location.href = "{{ url_for('goto', index=current_index - 1) }}";
                {% endif %}
            } else if (e.key === 'ArrowRight') {
                {% if current_index < total_count - 1 %}
                window.location.href = "{{ url_for('goto', index=current_index + 1) }}";
                {% endif %}
            }
        });
    </script>
    {% endif %}
</body>
</html>
'''


@app.route('/')
def index():
    """Main labeling interface."""
    current_index = progress['current_index']

    # Find next unlabeled
    next_unlabeled = current_index
    for i in range(len(intervention_pairs)):
        if i not in progress['labeled']:
            next_unlabeled = i
            break
    else:
        next_unlabeled = len(intervention_pairs)

    if current_index >= len(intervention_pairs):
        return render_template_string(
            HTML_TEMPLATE,
            current_index=current_index,
            total_count=len(intervention_pairs),
            labeled_count=len(progress['labeled']),
            progress_pct=100,
            output_file=OUTPUT_FILE
        )

    pair = intervention_pairs[current_index]
    doc_index = pair['doc_index']
    entry = documents[doc_index]
    n_comments = pair['comments_used']

    thread = [c.to_dict() for c in entry.comment_thread[:n_comments]]

    # Randomize display order (seeded by index for consistency when navigating)
    random.seed(current_index)
    swap_order = random.choice([True, False])

    if swap_order:
        # Left shows intervention_2, right shows intervention_1
        left_idx, right_idx = 2, 1
        left_text = pair['intervention_2']
        left_type = pair['intervention_2_type']
        right_text = pair['intervention_1']
        right_type = pair['intervention_1_type']
    else:
        # Left shows intervention_1, right shows intervention_2
        left_idx, right_idx = 1, 2
        left_text = pair['intervention_1']
        left_type = pair['intervention_1_type']
        right_text = pair['intervention_2']
        right_type = pair['intervention_2_type']

    return render_template_string(
        HTML_TEMPLATE,
        pair=pair,
        document=entry.document,
        highlighted_sentence=entry.highlighted_sentence,
        thread=thread,
        current_index=current_index,
        total_count=len(intervention_pairs),
        labeled_count=len(progress['labeled']),
        progress_pct=(len(progress['labeled']) / len(intervention_pairs)) * 100,
        is_labeled=current_index in progress['labeled'],
        next_unlabeled=next_unlabeled,
        output_file=OUTPUT_FILE,
        left_idx=left_idx,
        right_idx=right_idx,
        left_text=left_text,
        left_type=left_type,
        right_text=right_text,
        right_type=right_type
    )


@app.route('/goto/<int:index>')
def goto(index):
    """Navigate to a specific index."""
    progress['current_index'] = max(0, min(index, len(intervention_pairs)))
    save_progress()
    return redirect(url_for('index'))


@app.route('/label', methods=['POST'])
def label():
    """Handle labeling submission."""
    index = int(request.form['index'])
    choice = int(request.form['choice'])

    pair = intervention_pairs[index]
    doc_index = pair['doc_index']
    entry = documents[doc_index]
    n_comments = pair['comments_used']

    # Generate the prompt
    prompt = get_prompt_for_pair(pair, entry)

    # Determine accepted/rejected based on choice
    if choice == 0:
        # Skip - equal quality, don't save
        pass
    else:
        if choice == 1:
            accepted = pair['intervention_1']
            accepted_type = pair['intervention_1_type']
            rejected = pair['intervention_2']
            rejected_type = pair['intervention_2_type']
            accepted_agent = 1
        else:
            accepted = pair['intervention_2']
            accepted_type = pair['intervention_2_type']
            rejected = pair['intervention_1']
            rejected_type = pair['intervention_1_type']
            accepted_agent = 2

        result = {
            'doc_index': doc_index,
            'scenario': pair['scenario'],
            'comments_used': n_comments,
            'prompt': prompt,
            'accepted': accepted,
            'rejected': rejected,
            'accepted_type': accepted_type,
            'rejected_type': rejected_type,
            'accepted_agent': accepted_agent
        }

        save_label(result)

    # Update progress
    if index not in progress['labeled']:
        progress['labeled'].append(index)

    # Move to next
    progress['current_index'] = index + 1
    save_progress()

    return redirect(url_for('index'))


@app.route('/stats')
def stats():
    """Show labeling statistics."""
    labeled_indices = set(progress['labeled'])

    # Count by type for labeled pairs
    type_counts = {}
    for idx in labeled_indices:
        if idx < len(intervention_pairs):
            pair = intervention_pairs[idx]
            t1 = pair['intervention_1_type']
            t2 = pair['intervention_2_type']
            type_counts[t1] = type_counts.get(t1, 0) + 1
            type_counts[t2] = type_counts.get(t2, 0) + 1

    return jsonify({
        'total': len(intervention_pairs),
        'labeled': len(labeled_indices),
        'remaining': len(intervention_pairs) - len(labeled_indices),
        'type_counts': type_counts
    })


if __name__ == '__main__':
    load_data()
    print("\n" + "="*60)
    print("Manual Intervention Labeler")
    print("="*60)
    print(f"Open your browser to: http://127.0.0.1:5001")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print("="*60 + "\n")
    app.run(debug=True, port=5001)
