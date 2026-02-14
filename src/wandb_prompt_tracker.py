import os
import wandb
import json
from collections import defaultdict
from typing import Dict, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl

class PromptAccuracyTracker(TrainerCallback):
    """
    Tracks per-question accuracy over time and logs to wandb.
    
    NOTE: Wandb does NOT support programmatic creation of panels or dropdowns.
    However, this tracker logs data in a way that maximizes automatic visualization:
    - Uses wandb's metric grouping for automatic chart creation
    - Logs HTML visualizations that appear automatically
    - Creates tables that can be viewed in the Media tab
    
    Features:
    - Tracks accuracy per question_id (hash of question text) over time
    - Logs time series metrics for each question (auto-visualized by wandb)
    - Creates HTML visualizations with interactive dropdowns
    - Maintains a reference table of all questions seen
    """
    
    def __init__(self, recorder, log_table_every_n_steps: int = 1):
        """
        Args:
            recorder: RolloutRecorder instance that tracks the current step
            log_table_every_n_steps: How often to log tables/HTML (default: every step)
        """
        self.recorder = recorder
        self.log_table_every_n_steps = log_table_every_n_steps
        # Track question metadata (question_id -> question_text, first_seen_step)
        self.question_metadata = {}
        # Track all accuracy data over time for HTML visualization
        self.accuracy_history = {}  # {question_id: [(training_step, accuracy), ...]}
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called after each training step. Processes rollouts and logs to wandb.
        Using on_step_end instead of on_log ensures we catch every step.
        """
        if not wandb.run:
            return
            
        current_step = int(state.global_step)
        rollout_path = self.recorder.path
        
        # Initialize processed steps tracking
        if not hasattr(self, '_processed_steps'):
            self._processed_steps = set()
        
        try:
            # Check if file exists
            if not os.path.exists(rollout_path):
                return
            
            # Read ALL rollouts and find unprocessed ones
            all_records = []
            with open(rollout_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
            
            # Find records we haven't processed yet
            unprocessed_records = [
                r for r in all_records 
                if r.get("step") not in self._processed_steps
            ]
            
            if not unprocessed_records:
                return
            
            # Group unprocessed records by step
            records_by_step = defaultdict(list)
            for record in unprocessed_records:
                step = record.get("step")
                if step is not None:
                    records_by_step[step].append(record)
            
            # Process each step's records - calculate accuracy PER training step
            # Structure: {question_id: {training_step: {"correct": X, "total": Y}}}
            per_step_question_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
            all_question_stats = defaultdict(lambda: {"correct": 0, "total": 0, "question": "", "steps": []})
            
            for step, step_records in records_by_step.items():
                # step is the training step from rollouts (0, 1, 2, 3...)
                for record in step_records:
                    question_id = record.get("question_id")
                    if not question_id:
                        question_id = record.get("prompt_id")
                        if not question_id:
                            continue
                    
                    question_text = record.get("question", "")
                    is_correct = record.get("correct", False)
                    ground_truth = record.get("ground_truth", "")
                    
                    # Update metadata
                    if question_id not in self.question_metadata:
                        self.question_metadata[question_id] = {
                            "question": question_text,
                            "ground_truth": ground_truth,
                            "first_seen_step": step
                        }
                    else:
                        # Update question text if we didn't have it before
                        if not self.question_metadata[question_id]["question"] and question_text:
                            self.question_metadata[question_id]["question"] = question_text
                        # Update ground truth if we didn't have it before
                        if not self.question_metadata[question_id]["ground_truth"] and ground_truth:
                            self.question_metadata[question_id]["ground_truth"] = ground_truth
                    
                    # Track per-step stats for this question
                    step_stats = per_step_question_stats[question_id][step]
                    step_stats["total"] += 1
                    if is_correct:
                        step_stats["correct"] += 1
                    
                    # Also track overall stats
                    stats = all_question_stats[question_id]
                    stats["total"] += 1
                    if is_correct:
                        stats["correct"] += 1
                    if not stats["question"] and question_text:
                        stats["question"] = question_text
                    if step not in stats["steps"]:
                        stats["steps"].append(step)
                
                # Mark step as processed
                self._processed_steps.add(step)
            
            # Log metrics for each question (don't specify step - let wandb auto-increment)
            if all_question_stats:
                wandb_logs = {}
                for question_id, stats in all_question_stats.items():
                    accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                    question_text = stats["question"] or self.question_metadata.get(question_id, {}).get("question", "")
                    
                    # Log metric - wandb will auto-visualize these
                    wandb_logs[f"question_accuracy/{question_id}"] = accuracy
                    wandb_logs[f"question_count/{question_id}"] = stats["total"]
                    
                    # Track history for HTML visualization - use PER-STEP accuracy
                    if question_id not in self.accuracy_history:
                        self.accuracy_history[question_id] = []
                    # Calculate and store accuracy for each training step this question appeared in
                    for step in stats["steps"]:
                        # Get per-step stats for this question at this training step
                        step_stats = per_step_question_stats[question_id].get(step, {"correct": 0, "total": 0})
                        step_accuracy = step_stats["correct"] / step_stats["total"] if step_stats["total"] > 0 else 0.0
                        # Only add if we haven't already added this step
                        if not any(h[0] == step for h in self.accuracy_history[question_id]):
                            self.accuracy_history[question_id].append((step, step_accuracy))
                
                # Log aggregate stats
                all_correct = sum(s["correct"] for s in all_question_stats.values())
                all_total = sum(s["total"] for s in all_question_stats.values())
                overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
                wandb_logs["question_accuracy/mean"] = overall_accuracy
                wandb_logs["question_accuracy/num_questions"] = len(all_question_stats)
                
                # Log to wandb without specifying step - wandb will auto-increment
                # This avoids step ordering issues
                if wandb_logs:
                    wandb.log(wandb_logs)
                
                # Create HTML visualization periodically (every N steps)
                # This creates the interactive dropdown visualization
                if current_step % self.log_table_every_n_steps == 0:
                    self._log_html_visualization(current_step, all_question_stats)
                
                # Log question metadata as JSON artifact periodically
                if current_step % self.log_table_every_n_steps == 0 and self.question_metadata:
                    import tempfile
                    
                    # Create a JSON file with question metadata
                    metadata_file = {
                        "questions": {
                            qid: {
                                "question": meta["question"],
                                "ground_truth": meta.get("ground_truth", ""),
                                "first_seen_step": meta["first_seen_step"]
                            }
                            for qid, meta in self.question_metadata.items()
                        },
                        "current_step": current_step
                    }
                    
                    # Log as artifact (appears in Artifacts tab, downloadable)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(metadata_file, f, indent=2)
                        temp_path = f.name
                    
                    try:
                        artifact = wandb.Artifact(f"question_metadata_step_{current_step}", type="metadata")
                        artifact.add_file(temp_path, name="question_metadata.json")
                        wandb.log_artifact(artifact)
                    finally:
                        os.unlink(temp_path)
                        
        except Exception as e:
            # Log error but don't crash - print for debugging
            print(f"[PromptAccuracyTracker] Error at step {current_step}: {e}")
            import traceback
            traceback.print_exc()
    
    def _log_html_visualization(self, current_step: int, question_stats: dict):
        """Create an HTML visualization with three tabs: overall accuracy, question grid, and rollout viewer."""
        if not self.question_metadata:
            return
        
        # Load all rollout data for tab 3
        rollout_data = {}  # {question_id: {step: [rollouts]}}
        rollout_path = self.recorder.path
        if os.path.exists(rollout_path):
            with open(rollout_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        qid = record.get("question_id") or record.get("prompt_id")
                        step = record.get("step")
                        if qid and step is not None:
                            if qid not in rollout_data:
                                rollout_data[qid] = {}
                            if step not in rollout_data[qid]:
                                rollout_data[qid][step] = []
                            # Store relevant fields
                            rollout_data[qid][step].append({
                                "completion": record.get("completion", ""),
                                "prediction": record.get("prediction", ""),
                                "ground_truth": record.get("ground_truth", ""),
                                "correct": record.get("correct", False),
                                "reward": record.get("reward", 0.0)
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        # Build HTML with three tabs
        html_parts = []
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Question Accuracy Tracker</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }
                .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }
                .tab { padding: 12px 24px; cursor: pointer; background: #f5f5f5; border: none; border-bottom: 2px solid transparent; margin-right: 4px; font-size: 14px; font-weight: 500; color: #000; }
                .tab:hover { background: #e8e8e8; }
                .tab.active { background: white; border-bottom: 2px solid #4CAF50; color: #4CAF50; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                .chart-container { margin: 20px 0; }
                .question-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
                .question-box { padding: 15px; background: #f9f9f9; border: 2px solid #ddd; border-radius: 6px; cursor: pointer; transition: all 0.2s; }
                .question-box:hover { border-color: #4CAF50; box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2); }
                .question-box.selected { border-color: #4CAF50; background: #e8f5e9; }
                .question-preview { font-size: 13px; color: #555; margin-top: 8px; line-height: 1.4; }
                .question-preview strong { color: #333; }
                .question-detail { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 6px; border-left: 4px solid #4CAF50; }
                .question-detail h3 { margin-top: 0; color: #333; }
                .question-detail p { margin: 8px 0; line-height: 1.6; }
                .question-detail-chart { margin: 20px 0; }
                .controls { margin: 20px 0; }
                .controls label { display: block; margin-bottom: 8px; font-weight: 500; }
                .controls select { padding: 10px; font-size: 14px; width: 100%; max-width: 400px; border: 2px solid #ddd; border-radius: 4px; }
                .controls select:focus { border-color: #4CAF50; outline: none; }
                .rollout-list { margin: 20px 0; }
                .rollout-item { padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #ddd; background: #f9f9f9; }
                .rollout-item.correct { border-left-color: #4CAF50; background: #e8f5e9; }
                .rollout-item.incorrect { border-left-color: #f44336; background: #ffebee; }
                .rollout-item strong { display: block; margin-bottom: 8px; }
                .rollout-completion { margin-top: 8px; padding: 10px; background: white; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
                .rollout-meta { margin-top: 8px; font-size: 12px; color: #666; }
                .rollout-meta span { margin-right: 20px; }
                .rollout-meta span:last-child { margin-right: 0; }
                h2 { color: #333; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📊 Per-Question Accuracy Over Time</h2>
                <div class="tabs">
                    <button class="tab active" onclick="switchTab(0)">📈 Overall Accuracy</button>
                    <button class="tab" onclick="switchTab(1)">📋 Question Browser</button>
                    <button class="tab" onclick="switchTab(2)">🔍 Rollout Viewer</button>
                </div>
                
                <!-- Tab 1: Overall Accuracy -->
                <div id="tab0" class="tab-content active">
                    <div class="chart-container">
                        <div id="overallChart"></div>
                    </div>
                </div>
                
                <!-- Tab 2: Question Browser -->
                <div id="tab1" class="tab-content">
                    <div id="questionDetail" class="question-detail" style="display: none;"></div>
                    <div class="question-grid" id="questionGrid"></div>
                </div>
                
                <!-- Tab 3: Rollout Viewer -->
                <div id="tab2" class="tab-content">
                    <div class="controls">
                        <label for="rolloutQuestionSelect">Question ID:</label>
                        <select id="rolloutQuestionSelect" onchange="updateRolloutStepDropdown()">
                            <option value="">-- Select Question --</option>
                        </select>
                    </div>
                    <div class="controls">
                        <label for="rolloutStepSelect">Training Step:</label>
                        <select id="rolloutStepSelect" onchange="displayRollouts()">
                            <option value="">-- Select Step --</option>
                        </select>
                    </div>
                    <div id="rolloutChartContainer" class="chart-container" style="display: none;">
                        <div id="rolloutChart"></div>
                    </div>
                    <div id="rolloutList" class="rollout-list"></div>
                </div>
            </div>
            
            <script>
                const accuracyData = {
        """)
        
        # Add data for each question
        for question_id, history in self.accuracy_history.items():
            sorted_history = sorted(history, key=lambda x: x[0])
            steps = [h[0] for h in sorted_history]
            accuracies = [h[1] for h in sorted_history]
            html_parts.append(f'"{question_id}": {{')
            html_parts.append(f'  steps: {json.dumps(steps)},')
            html_parts.append(f'  accuracies: {json.dumps(accuracies)}')
            html_parts.append('},')
        
        # Calculate average across all questions
        if self.accuracy_history:
            all_steps = set()
            for history in self.accuracy_history.values():
                all_steps.update([h[0] for h in history])
            
            avg_data = {}
            for step in sorted(all_steps):
                accs = []
                for history in self.accuracy_history.values():
                    for s, acc in history:
                        if s == step:
                            accs.append(acc)
                if accs:
                    avg_data[step] = sum(accs) / len(accs)
            
            html_parts.append('"all": {')
            html_parts.append(f'  steps: {json.dumps(sorted(avg_data.keys()))},')
            html_parts.append(f'  accuracies: {json.dumps([avg_data[s] for s in sorted(avg_data.keys())])}')
            html_parts.append('}')
        
        html_parts.append("""
                };
                
                const questionMetadata = {
        """)
        
        # Add question metadata
        for question_id, meta in self.question_metadata.items():
            html_parts.append(f'"{question_id}": {{')
            html_parts.append(f'  question: {json.dumps(meta.get("question", ""))},')
            html_parts.append(f'  ground_truth: {json.dumps(meta.get("ground_truth", ""))},')
            html_parts.append(f'  first_seen_step: {meta.get("first_seen_step", 0)}')
            html_parts.append('},')
        
        html_parts.append("""
                };
                
                const rolloutData = {
        """)
        
        # Add rollout data
        for question_id, step_data in rollout_data.items():
            html_parts.append(f'"{question_id}": {{')
            for step, rollouts in step_data.items():
                html_parts.append(f'  "{step}": {json.dumps(rollouts)},')
            html_parts.append('},')
        
        html_parts.append("""
                };
                
                function switchTab(index) {
                    // Hide all tabs
                    for (let i = 0; i < 3; i++) {
                        document.getElementById(`tab${i}`).classList.remove('active');
                        document.querySelectorAll('.tab')[i].classList.remove('active');
                    }
                    // Show selected tab
                    document.getElementById(`tab${index}`).classList.add('active');
                    document.querySelectorAll('.tab')[index].classList.add('active');
                    
                    // Initialize tab content if needed
                    if (index === 0) {
                        updateOverallChart();
                    } else if (index === 1) {
                        renderQuestionGrid();
                    } else if (index === 2) {
                        populateRolloutQuestionDropdown();
                    }
                }
                
                // Helper function to calculate dynamic x-axis tick spacing (max 32 ticks)
                function getXAxisConfig(steps) {
                    if (!steps || steps.length === 0) {
                        return { tickmode: 'linear', dtick: 1 };
                    }
                    
                    const minStep = Math.min(...steps);
                    const maxStep = Math.max(...steps);
                    const stepRange = maxStep - minStep;
                    
                    if (stepRange === 0) {
                        return { tickmode: 'linear', dtick: 1 };
                    }
                    
                    // Calculate tick spacing to have at most 32 ticks
                    const maxTicks = 32;
                    const tickSpacing = Math.ceil(stepRange / maxTicks);
                    
                    return {
                        tickmode: 'linear',
                        dtick: tickSpacing,
                        title: 'Training Step (1, 2, 3, ...)',
                        gridcolor: '#e0e0e0'
                    };
                }
                
                // Helper function to calculate dynamic y-axis range
                function getYAxisConfig(accuracies) {
                    if (!accuracies || accuracies.length === 0) {
                        return { title: 'Accuracy', range: [0, 1], gridcolor: '#e0e0e0' };
                    }
                    
                    const minAcc = Math.min(...accuracies);
                    const maxAcc = Math.max(...accuracies);
                    const accRange = maxAcc - minAcc;
                    
                    // Add padding: 10% on each side, but ensure we don't go below 0 or above 1
                    const padding = Math.max(accRange * 0.1, 0.05);
                    let yMin = Math.max(0, minAcc - padding);
                    let yMax = Math.min(1, maxAcc + padding);
                    
                    // If the range is very small, ensure we have at least some visible range
                    if (yMax - yMin < 0.1) {
                        const center = (yMin + yMax) / 2;
                        yMin = Math.max(0, center - 0.1);
                        yMax = Math.min(1, center + 0.1);
                    }
                    
                    // If all values are very close to 0 or 1, use a reasonable default
                    if (yMax - yMin < 0.01) {
                        if (maxAcc < 0.1) {
                            yMin = 0;
                            yMax = 0.2;
                        } else if (minAcc > 0.9) {
                            yMin = 0.8;
                            yMax = 1;
                        } else {
                            yMin = 0;
                            yMax = 1;
                        }
                    }
                    
                    return {
                        title: 'Accuracy',
                        range: [yMin, yMax],
                        gridcolor: '#e0e0e0'
                    };
                }
                
                function updateOverallChart() {
                    const allData = accuracyData["all"];
                    const trace = {
                        x: allData.steps,
                        y: allData.accuracies,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Average Accuracy',
                        line: { width: 3, color: '#4CAF50' },
                        marker: { size: 6 }
                    };
                    
                    const layout = {
                        title: 'Overall Accuracy Over Time (All Questions)',
                        xaxis: getXAxisConfig(allData.steps),
                        yaxis: getYAxisConfig(allData.accuracies),
                        hovermode: 'closest',
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white'
                    };
                    
                    Plotly.newPlot('overallChart', [trace], layout);
                }
                
                function renderQuestionGrid() {
                    const grid = document.getElementById('questionGrid');
                    grid.innerHTML = '';
                    
                    const questions = Object.keys(questionMetadata).sort((a, b) => {
                        return questionMetadata[a].first_seen_step - questionMetadata[b].first_seen_step;
                    });
                    
                    questions.forEach(qid => {
                        const meta = questionMetadata[qid];
                        const questionText = meta.question || 'No question text';
                        const preview = questionText.length > 100 ? questionText.substring(0, 100) + '...' : questionText;
                        
                        const box = document.createElement('div');
                        box.className = 'question-box';
                        box.onclick = function() { showQuestionDetail(qid, this); };
                        box.innerHTML = `
                            <strong>ID: ${qid.substring(0, 12)}...</strong>
                            <div class="question-preview">${preview}</div>
                        `;
                        grid.appendChild(box);
                    });
                }
                
                function showQuestionDetail(questionId, clickedBox) {
                    const meta = questionMetadata[questionId];
                    const detailDiv = document.getElementById('questionDetail');
                    
                    // Update selected box
                    document.querySelectorAll('.question-box').forEach(box => {
                        box.classList.remove('selected');
                    });
                    if (clickedBox) {
                        clickedBox.classList.add('selected');
                    }
                    
                    // Get accuracy data for this question
                    const questionData = accuracyData[questionId];
                    const hasData = questionData && questionData.steps && questionData.steps.length > 0;
                    
                    detailDiv.style.display = 'block';
                    detailDiv.innerHTML = `
                        <h3>Question Details</h3>
                        <p><strong>Question ID:</strong> ${questionId}</p>
                        <p><strong>Question:</strong> ${meta.question || 'N/A'}</p>
                        <p><strong>Correct Answer:</strong> ${meta.ground_truth || 'N/A'}</p>
                        <p><strong>First Seen:</strong> Training Step ${meta.first_seen_step}</p>
                        ${hasData ? '<div class="question-detail-chart"><div id="questionDetailChart"></div></div>' : '<p><em>No accuracy data available yet for this question.</em></p>'}
                    `;
                    
                    // Render chart if data is available
                    if (hasData) {
                        setTimeout(() => {
                            const trace = {
                                x: questionData.steps,
                                y: questionData.accuracies,
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: 'Accuracy',
                                line: { width: 3, color: '#2196F3' },
                                marker: { size: 6 }
                            };
                            
                            const layout = {
                                title: 'Accuracy Over Time',
                                xaxis: getXAxisConfig(questionData.steps),
                                yaxis: getYAxisConfig(questionData.accuracies),
                                hovermode: 'closest',
                                plot_bgcolor: '#fafafa',
                                paper_bgcolor: 'white',
                                height: 400
                            };
                            
                            Plotly.newPlot('questionDetailChart', [trace], layout);
                        }, 100);
                    }
                }
                
                function populateRolloutQuestionDropdown() {
                    const select = document.getElementById('rolloutQuestionSelect');
                    select.innerHTML = '<option value="">-- Select Question --</option>';
                    
                    Object.keys(rolloutData).forEach(qid => {
                        const meta = questionMetadata[qid];
                        const questionText = meta.question || 'No question';
                        const display = questionText.length > 60 ? questionText.substring(0, 60) + '...' : questionText;
                        const option = document.createElement('option');
                        option.value = qid;
                        option.textContent = `${qid.substring(0, 12)}... - ${display}`;
                        select.appendChild(option);
                    });
                }
                
                function updateRolloutStepDropdown() {
                    const questionSelect = document.getElementById('rolloutQuestionSelect');
                    const stepSelect = document.getElementById('rolloutStepSelect');
                    const questionId = questionSelect.value;
                    const chartContainer = document.getElementById('rolloutChartContainer');
                    
                    stepSelect.innerHTML = '<option value="">-- Select Step --</option>';
                    
                    if (questionId && rolloutData[questionId]) {
                        const steps = Object.keys(rolloutData[questionId]).map(s => parseInt(s)).sort((a, b) => a - b);
                        steps.forEach(step => {
                            const option = document.createElement('option');
                            option.value = step;
                            option.textContent = `Step ${step}`;
                            stepSelect.appendChild(option);
                        });
                        
                        // Show and update accuracy chart
                        updateRolloutChart(questionId);
                    } else {
                        // Hide chart if no question selected
                        chartContainer.style.display = 'none';
                    }
                    
                    displayRollouts();
                }
                
                function updateRolloutChart(questionId) {
                    const chartContainer = document.getElementById('rolloutChartContainer');
                    const questionData = accuracyData[questionId];
                    
                    if (!questionData || !questionData.steps || questionData.steps.length === 0) {
                        chartContainer.style.display = 'none';
                        return;
                    }
                    
                    chartContainer.style.display = 'block';
                    
                    setTimeout(() => {
                        const trace = {
                            x: questionData.steps,
                            y: questionData.accuracies,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Accuracy',
                            line: { width: 3, color: '#2196F3' },
                            marker: { size: 6 }
                        };
                        
                        const layout = {
                            title: 'Accuracy Over Time',
                            xaxis: getXAxisConfig(questionData.steps),
                            yaxis: getYAxisConfig(questionData.accuracies),
                            hovermode: 'closest',
                            plot_bgcolor: '#fafafa',
                            paper_bgcolor: 'white',
                            height: 400
                        };
                        
                        Plotly.newPlot('rolloutChart', [trace], layout);
                    }, 100);
                }
                
                function displayRollouts() {
                    const questionSelect = document.getElementById('rolloutQuestionSelect');
                    const stepSelect = document.getElementById('rolloutStepSelect');
                    const questionId = questionSelect.value;
                    const step = stepSelect.value;
                    const listDiv = document.getElementById('rolloutList');
                    
                    listDiv.innerHTML = '';
                    
                    if (!questionId) {
                        return;
                    }
                    
                    // Update chart if question is selected (even if step isn't)
                    if (questionId) {
                        updateRolloutChart(questionId);
                    }
                    
                    if (!step) {
                        return;
                    }
                    
                    const rollouts = rolloutData[questionId] && rolloutData[questionId][step];
                    if (!rollouts || rollouts.length === 0) {
                        listDiv.innerHTML = '<p>No rollouts found for this question at this step.</p>';
                        return;
                    }
                    
                    rollouts.forEach((rollout, idx) => {
                        const item = document.createElement('div');
                        item.className = `rollout-item ${rollout.correct ? 'correct' : 'incorrect'}`;
                        item.innerHTML = `
                            <strong>Rollout ${idx + 1} - ${rollout.correct ? '✓ Correct' : '✗ Incorrect'}</strong>
                            <div class="rollout-meta">
                                <span><strong>Prediction:</strong> ${rollout.prediction || 'N/A'}</span>
                                <span><strong>Ground Truth:</strong> ${rollout.ground_truth || 'N/A'}</span>
                                <span><strong>Reward:</strong> ${rollout.reward.toFixed(2)}</span>
                            </div>
                            <div class="rollout-completion">${rollout.completion || 'N/A'}</div>
                        `;
                        listDiv.appendChild(item);
                    });
                }
                
                // Initialize
                updateOverallChart();
            </script>
        </body>
        </html>
        """)
        
        html_content = "".join(html_parts)
        
        # Log as HTML without step parameter (let wandb auto-increment)
        # This appears in Files > media > html folder
        wandb.log({"question_accuracy_interactive": wandb.Html(html_content)})
