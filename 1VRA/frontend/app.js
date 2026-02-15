const API_URL = 'http://localhost:8000';

const fileInput = document.getElementById('csvFile');
const fileNameDisplay = document.getElementById('fileName');
const calculateBtn = document.getElementById('calculateBtn');
const serverLogs = document.getElementById('serverLogs');
const previewCard = document.getElementById('previewCard');
const previewTable = document.getElementById('previewTable');
const outputPathDisplay = document.getElementById('outputPath');
// Volatility Elements
const volInputPath = document.getElementById('volInputPath');
const volCsvFile = document.getElementById('volCsvFile');
const volWindowInput = document.getElementById('volWindow');
const computeVolBtn = document.getElementById('computeVolBtn');
const volServerLogs = document.getElementById('volServerLogs');
const volPreviewCard = document.getElementById('volPreviewCard');
const volPreviewTable = document.getElementById('volPreviewTable');
const volOutputPath = document.getElementById('volOutputPath');
const volParamsList = document.getElementById('volParamsList');
// Risk Labelling Elements
const labelStrategy = document.getElementById('labelStrategy');
const labelDesc = document.getElementById('labelDesc');
const labelBtn = document.getElementById('labelBtn');

let pollingInterval = null;
let selectedVolFile = null;

// File Selection
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.style.color = 'var(--text-main)';
        calculateBtn.disabled = false;
    } else {
        fileNameDisplay.textContent = 'No file selected';
        fileNameDisplay.style.color = 'var(--text-muted)';
        calculateBtn.disabled = true;
    }
});

// Tab Switching
document.querySelectorAll('.nav-links li').forEach(li => {
    li.addEventListener('click', () => {
        if (li.classList.contains('disabled')) return;

        // Update Nav
        document.querySelectorAll('.nav-links li').forEach(l => l.classList.remove('active'));
        li.classList.add('active');

        // Show Content
        const tabId = li.getAttribute('data-tab');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
        document.getElementById(tabId).classList.remove('hidden');

        // Specific Logic
        if (tabId === 'volatility') {
            // Auto-fill path from previous step if available and input is empty
            if (outputPathDisplay.textContent !== '...' && !volInputPath.value) {
                volInputPath.value = outputPathDisplay.textContent;
            }
        }
    });
});

// Volatility File Browse
volCsvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedVolFile = file;
        volInputPath.value = `[Upload] ${file.name}`;
    }
});

// Calculate Button Click
calculateBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Reset UI
    calculateBtn.disabled = true;
    calculateBtn.textContent = 'Processing...';
    previewCard.classList.add('hidden');
    serverLogs.innerHTML = '';

    addLog('system', 'Uploading file to server...');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            addLog('info', `Upload success. Server processing: ${data.filename}`);
            startPolling();
        } else {
            addLog('error', 'Upload failed.');
            calculateBtn.disabled = false;
            calculateBtn.textContent = 'Calculate Indicators';
        }
    } catch (error) {
        addLog('error', `Connection error: ${error.message}`);
        calculateBtn.disabled = false;
        calculateBtn.textContent = 'Calculate Indicators';
    }
});

// Volatility Compute Button
computeVolBtn.addEventListener('click', async () => {
    let filePath = volInputPath.value;
    const windowSize = volWindowInput.value;

    // Reset UI
    computeVolBtn.disabled = true;
    computeVolBtn.textContent = 'Processing...';
    volPreviewCard.classList.add('hidden');
    volServerLogs.innerHTML = '';

    // 1. Handle File Upload if needed
    if (selectedVolFile) {
        addLog(volServerLogs, 'system', 'Uploading new file...');
        const formData = new FormData();
        formData.append('file', selectedVolFile);

        try {
            const uploadResp = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            if (!uploadResp.ok) throw new Error('Upload failed');
            const data = await uploadResp.json();
            // We need the full path. The upload endpoint returns filename, maybe we can reconstruct path or update upload endpoint?
            // Wait, the current logic relies on backend `processing_status["last_file"]` or output path.
            // But `/upload` endpoint returns `filename`. The backend stores it in `DATA_DIR`.
            // I need the full absolute path for `/calculate-volatility`.
            // Let's assume the backend handles relative filename in `DATA_DIR` if absolute path not provided?
            // Actually, I'll update `/upload` or just guess the path? 
            // The logs show the path: "File uploaded to ...". 
            // Better: update `/calculate-volatility` to accept just a filename and look in data dir?
            // OR: Just rely on `process_file_task` updating status.
            // Actually, let's use the response filename and let backend resolve it.
            // I will update the InputPath with the filename, but I need the full path for sure?
            // The `calculate_volatility` endpoint expects `file_path`.

            // Let's Peek at `upload_file` in main.py:
            // `file_path = DATA_DIR / filename`
            // It doesn't return the full path in JSON.
            // But `processing_status["last_file"]` matches filename.

            // Re-reading main.py: `calculate_volatility` creates `file_path_obj = Path(file_path)`.
            // If I send just filename, `Path("filename").exists()` checks CWD. That might fail.
            // I should modify `calculate_volatility` or `upload` to return path.

            // Hack for now: I know the `DATA_DIR` structure is `../data`.
            // But javascript doesn't know absolute server path.
            // I will Assume the backend `file_path` arg can handle filenames in DATA_DIR if I modify backend?
            // NO, I can't modify backend again easily without checking.
            // Wait, I updated `main.py` recently.

            // Let's modify `main.py`'s `upload_file` return to include `file_path` string?
            // Actually, I didn't modify `upload_file` return in previous step.

            // Safe bet: The user's prompt says "The file generated by indicators calculor should be preselected".
            // That provides a full path in `outputPathDisplay`.
            // If they upload a NEW file, maybe I should just use the `upload` endpoint which TRIGGERS `process_file_task` automatically?
            // Wait, `upload_file` in `main.py` adds `process_file_task` to background tasks!
            // So if I upload a file, it AUTOMATICALLY runs the Indicators calculation first.
            // That might be desired? "The file generated by indicators calculor should be preselected"
            // If user uploads a RAW csv, do they want to skip indicators and go straight to Volatility?
            // Prompt says: "compute volatility... The file generated by indicators calculor should be preselected but the user should have the option for other file as well".
            // If they pick "other file", it might be a raw one or a pre-processed one.
            // If raw, it needs `intc` etc.
            // If I upload via `/upload`, it runs indicators. That produces a `PROCESSED_` file.
            // So I should wait for that to finish, THEN run Volatility on the result?
            // That seems like a solid workflow.

            // So: If upload -> Wait for Indicators -> Then run Volatility on the output.
            // That's complex for JS.

            // Option 2: Just send the file to a new pure-upload endpoint? 
            // No, I shouldn't change backend too much.

            // Let's assume the user selects a file that is ALREADY on the server (via path input) OR they upload a file that goes through the standard pipeline.
            // If they upload, I'll let the standard pipeline run, and tell them "File uploaded and processing standard indicators. Please wait..."
            // THEN they can run Volatility?
            // Or I can chain it.

            // Simplest path: If file uploaded, just run the standard upload.
            // Then fill the input with the OUTPUT of that.
            // THEN click Compute Volatility.

            // But the button says "Compute Volatility".
            // If I implement "Upload -> Wait -> Volatility" that's best.

            // Implementation:
            // 1. Upload.
            // 2. Poll Generic Status until 'completed'.
            // 3. Get output path.
            // 4. Call `calculate-volatility` with that path.

            addLog(volServerLogs, 'info', 'File uploaded. Processing base indicators first...');
            await pollGenericForCompletion();
            // Check status for output file
            const statusResp = await fetch(`${API_URL}/status`);
            const status = await statusResp.json();
            filePath = status.output_file;

            selectedVolFile = null; // Clear selection

        } catch (e) {
            addLog(volServerLogs, 'error', `Upload error: ${e.message}`);
            computeVolBtn.disabled = false;
            computeVolBtn.textContent = 'Compute Volatility';
            return;
        }
    }

    if (!filePath) {
        addLog(volServerLogs, 'error', 'No file path provided.');
        computeVolBtn.disabled = false;
        computeVolBtn.textContent = 'Compute Volatility';
        return;
    }

    addLog(volServerLogs, 'system', `Requesting volatility for ${filePath}...`);

    // Call Calculate Volatility
    try {
        // Need to encode params?
        const url = new URL(`${API_URL}/calculate-volatility`);
        url.searchParams.append('file_path', filePath);
        url.searchParams.append('window', windowSize);

        const resp = await fetch(url, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Request failed');
        }

        startVolPolling();

    } catch (error) {
        addLog(volServerLogs, 'error', `Error: ${error.message}`);
        computeVolBtn.disabled = false;
        computeVolBtn.textContent = 'Compute Volatility';
    }
});

// Risk Labeling Logic
labelStrategy.addEventListener('change', () => {
    const val = labelStrategy.value;
    if (val === 'relative_thresholds') {
        labelDesc.value = `Compare each row’s volatility to the historical distribution (entire file).\nLow risk = Vol < 33rd percentile\nMedium risk = 33rd - 66th percentile\nHigh risk = > 66th percentile`;
    } else {
        labelDesc.value = 'Strategy details unavailable.';
    }
});

labelBtn.addEventListener('click', async () => {
    // Determine input file path
    // We should use the OUTPUT of the Volatility calculation as the input for Labeling.
    // If user hasn't computed Volatility yet, warning?
    // Or check volOutputPath.textContent?
    let filePath = volOutputPath.textContent;

    if (!filePath || filePath === '...') {
        // Fallback: Check if they input a path manually in the first section?
        if (volInputPath.value && volInputPath.value.includes('VOL_')) {
            filePath = volInputPath.value;
        } else {
            addLog(volServerLogs, 'error', 'Please compute volatility or ensure a volatility file is selected first.');
            return;
        }
    }

    const strategy = labelStrategy.value;

    // Reset UI
    labelBtn.disabled = true;
    labelBtn.textContent = 'Labelling...';
    volPreviewCard.classList.add('hidden');
    volServerLogs.innerHTML = '';

    addLog(volServerLogs, 'system', `Starting Risk Labelling on ${filePath}...`);

    try {
        const url = new URL(`${API_URL}/apply-labels`);
        url.searchParams.append('file_path', filePath);
        url.searchParams.append('strategy', strategy);

        const resp = await fetch(url, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Request failed');
        }

        // Use standard poller but need to update the button at the end
        // I can reuse startVolPolling because it updates `computeVolBtn`... 
        // Wait, `startVolPolling` specifically references `computeVolBtn`.
        // I should refactor `startVolPolling` to be more generic OR create `startLabelPolling`.
        // Cleaner to create a specific one or pass callback. 
        // For speed, I'll copy-paste-modify to `startLabelPolling`.

        startLabelPolling();

    } catch (error) {
        addLog(volServerLogs, 'error', `Error: ${error.message}`);
        labelBtn.disabled = false;
        labelBtn.textContent = 'Start Labelling';
    }
});

function startLabelPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            volServerLogs.innerHTML = '';
            addLog(volServerLogs, 'info', `Status: ${status.status}`);
            status.details.forEach(detail => {
                let type = 'info';
                if (detail.includes('Error')) type = 'error';
                else if (detail.includes('Complete')) type = 'success';

                const div = document.createElement('div');
                div.className = `log-entry ${type}`;
                div.textContent = detail;
                volServerLogs.appendChild(div);
            });
            volServerLogs.scrollTop = volServerLogs.scrollHeight;

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                labelBtn.disabled = false;
                labelBtn.textContent = 'Start Labelling';
                addLog(volServerLogs, 'success', 'Labelling Finished!');
                volOutputPath.textContent = status.output_file;

                // Show params update
                const currentParams = volParamsList.innerHTML;
                if (!currentParams.includes('Strategy')) {
                    volParamsList.innerHTML += `<li><strong>Labelling:</strong> ${labelStrategy.options[labelStrategy.selectedIndex].text}</li>`;
                }

                loadPreview(volPreviewTable, volPreviewCard, true);
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                labelBtn.disabled = false;
                labelBtn.textContent = 'Retry';
                addLog(volServerLogs, 'error', 'Stopped due to error.');
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}
// Modified addLog to accept target container
function addLog(container, type, message) {
    // legacy support if first arg is string (old calls)
    if (typeof container === 'string') {
        message = type;
        type = container;
        container = serverLogs;
    }

    // Safety check for container
    if (!container) return;

    const div = document.createElement('div');
    div.className = `log-entry ${type}`;
    div.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// Polling for Volatility
function startVolPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            volServerLogs.innerHTML = '';
            addLog(volServerLogs, 'info', `Status: ${status.status}`);
            status.details.forEach(detail => {
                let type = 'info';
                if (detail.includes('Error')) type = 'error';
                else if (detail.includes('Complete')) type = 'success';

                const div = document.createElement('div');
                div.className = `log-entry ${type}`;
                div.textContent = detail;
                volServerLogs.appendChild(div);
            });
            volServerLogs.scrollTop = volServerLogs.scrollHeight;

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                computeVolBtn.disabled = false;
                computeVolBtn.textContent = 'Compute Volatility';
                addLog(volServerLogs, 'success', 'Computation Finished!');
                volOutputPath.textContent = status.output_file;

                // Show params
                volParamsList.innerHTML = `<li><strong>Window Size:</strong> ${volWindowInput.value}</li>`;

                loadPreview(volPreviewTable, volPreviewCard, true);
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                computeVolBtn.disabled = false;
                computeVolBtn.textContent = 'Retry';
                addLog(volServerLogs, 'error', 'Stopped due to error.');
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            // We use the 'serverLogs' container for the main tab
            serverLogs.innerHTML = '';
            addLog(serverLogs, 'info', `Status: ${status.status}`);
            status.details.forEach(detail => {
                let type = 'info';
                if (detail.includes('Error')) type = 'error';
                else if (detail.includes('Complete')) type = 'success';
                const div = document.createElement('div');
                div.className = `log-entry ${type}`;
                div.textContent = detail;
                serverLogs.appendChild(div);
            });
            serverLogs.scrollTop = serverLogs.scrollHeight;

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                calculateBtn.disabled = false;
                calculateBtn.textContent = 'Calculate Indicators';
                addLog(serverLogs, 'success', 'Processing finished!');
                outputPathDisplay.textContent = status.output_file;
                loadPreview(previewTable, previewCard, false);
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                calculateBtn.disabled = false;
                calculateBtn.textContent = 'Retry';
                addLog(serverLogs, 'error', 'Processing stopped due to error.');
            }
        } catch (e) { console.error(e); }
    }, 1000);
}

async function loadPreview(tableEl, cardEl, isVol) {
    // Default to main elements if not provided (legacy)
    if (!tableEl) tableEl = previewTable;
    if (!cardEl) cardEl = previewCard;

    try {
        const response = await fetch(`${API_URL}/preview`);
        const data = await response.json();

        if (data.error) {
            // only log error to specific container if possible
            const logDiv = document.createElement('div');
            logDiv.className = 'log-entry error';
            logDiv.textContent = data.error;

            if (cardEl === volPreviewCard) volServerLogs.appendChild(logDiv);
            else serverLogs.appendChild(logDiv);

            return;
        }

        // Render Table Headers
        const thead = tableEl.querySelector('thead');
        thead.innerHTML = '';
        const headerRow = document.createElement('tr');
        data.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Render Body
        const tbody = tableEl.querySelector('tbody');
        tbody.innerHTML = '';
        data.preview.forEach(row => {
            const tr = document.createElement('tr');
            data.columns.forEach(col => {
                const td = document.createElement('td');
                let val = row[col];

                // Format numbers
                if (typeof val === 'number') {
                    val = val.toFixed(4);
                } else if (val === null) {
                    val = '-';
                }

                td.textContent = val;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });

        cardEl.classList.remove('hidden');

    } catch (error) {
        console.error(error);
    }
}

function copyPath(elementId) {
    // If no ID provided, fallback to default (old behavior)
    const el = document.getElementById(elementId) || outputPathDisplay;
    const path = el.textContent;
    navigator.clipboard.writeText(path).then(() => {
        alert('Path copied to clipboard!');
    });
}
