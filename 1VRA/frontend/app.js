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
// Clustering Elements
const clusterInputPath = document.getElementById('clusterInputPath');
const clusterCsvFile = document.getElementById('clusterCsvFile');
const paramN = document.getElementById('paramN');
const paramK = document.getElementById('paramK');
const clusterBtn = document.getElementById('clusterBtn');
const clusterServerLogs = document.getElementById('clusterServerLogs');
const clusterResultsList = document.getElementById('clusterResultsList');
const clusterPreviewCard = document.getElementById('clusterPreviewCard');
const clusterPreviewTable = document.getElementById('clusterPreviewTable');
const clusterOutputPath = document.getElementById('clusterOutputPath');
const elbowChartCard = document.getElementById('elbowChartCard');
const elbowChartCanvas = document.getElementById('elbowChart');
const clusterAlgorithm = document.getElementById('clusterAlgorithm');
const dbscanParams = document.getElementById('dbscanParams');
const kmeansParams = document.getElementById('kmeansParams');
const kmeansK = document.getElementById('kmeansK');
const kmeansKMin = document.getElementById('kmeansKMin');
const kmeansKMax = document.getElementById('kmeansKMax');

let pollingInterval = null;
let selectedVolFile = null;
let selectedClusterFile = null;
let elbowChart = null;  // Chart.js instance

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
        } else if (tabId === 'clustering') {
            // Auto-fill from Volatility/Label output
            if (volOutputPath.textContent !== '...' && !clusterInputPath.value) {
                clusterInputPath.value = volOutputPath.textContent;
            }
        }
    });
});



// Algorithm Switcher
clusterAlgorithm.addEventListener('change', () => {
    const algo = clusterAlgorithm.value;
    if (algo === 'dbscan') {
        dbscanParams.classList.remove('hidden');
        kmeansParams.classList.add('hidden');
    } else if (algo === 'kmeans') {
        dbscanParams.classList.add('hidden');
        kmeansParams.classList.remove('hidden');
    }
});

// Volatility File Browse
volCsvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedVolFile = file;
        volInputPath.value = `[Upload] ${file.name}`;
    }
});

// Clustering File Browse
clusterCsvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedClusterFile = file;
        clusterInputPath.value = `[Upload] ${file.name}`;
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

// Clustering Logic
clusterBtn.addEventListener('click', async () => {
    let filePath = clusterInputPath.value;
    const algorithm = clusterAlgorithm.value;

    // Check for file upload logic (similar to previously implemented)
    if (selectedClusterFile) {
        addLog(clusterServerLogs, 'info', 'Uploading clustering file...');
        const formData = new FormData();
        formData.append('file', selectedClusterFile);
        try {
            const upResp = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            if (!upResp.ok) throw new Error('Upload failed');
            // Wait for processing...
            addLog(clusterServerLogs, 'info', 'Processing features (indicators)...');
            await pollGenericForCompletion();
            // Get path
            const s = await (await fetch(`${API_URL}/status`)).json();
            filePath = s.output_file;
            selectedClusterFile = null;
        } catch (e) {
            addLog(clusterServerLogs, 'error', 'Upload failed: ' + e.message);
            return;
        }
    } else if (!filePath || filePath === '...') {
        // Fallback
        filePath = volOutputPath.textContent;
        if (!filePath || filePath === '...') {
            addLog(clusterServerLogs, 'error', 'No input file specified.');
            return;
        }
    }

    if (filePath.startsWith('[Upload]')) {
        // Didnt finish upload logic above? 
        // Should happen inside selectedClusterFile check.
    }

    // Reset UI
    clusterBtn.disabled = true;
    clusterBtn.textContent = 'Clustering...';
    clusterPreviewCard.classList.add('hidden');
    elbowChartCard.classList.add('hidden');
    clusterServerLogs.innerHTML = '';

    try {
        let url, logMsg;

        if (algorithm === 'dbscan') {
            const n = paramN.value;
            const k = paramK.value;
            logMsg = `Starting DBSCAN (n=${n}, k=${k}) on ${filePath}...`;

            url = new URL(`${API_URL}/train-dbscan`);
            url.searchParams.append('file_path', filePath);
            url.searchParams.append('n', n);
            url.searchParams.append('k', k);

        } else if (algorithm === 'kmeans') {
            const kVal = kmeansK.value;
            const kMin = kmeansKMin.value;
            const kMax = kmeansKMax.value;

            logMsg = kVal === 'auto' || kVal === ''
                ? `Starting K-Means (auto K, range ${kMin}-${kMax}) on ${filePath}...`
                : `Starting K-Means (K=${kVal}) on ${filePath}...`;

            url = new URL(`${API_URL}/train-kmeans`);
            url.searchParams.append('file_path', filePath);
            url.searchParams.append('k', kVal === 'auto' || kVal === '' ? 'auto' : kVal);
            url.searchParams.append('k_min', kMin);
            url.searchParams.append('k_max', kMax);
        }

        addLog(clusterServerLogs, 'system', logMsg);

        const resp = await fetch(url, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Request failed');
        }

        startClusterPolling();

    } catch (error) {
        addLog(clusterServerLogs, 'error', `Error: ${error.message}`);
        clusterBtn.disabled = false;
        clusterBtn.textContent = 'Run Clustering';
    }
});

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

// Helper: Generic Poller Promise
function pollGenericForCompletion() {
    return new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
            try {
                const r = await fetch(`${API_URL}/status`);
                const s = await r.json();
                if (s.status === 'completed') {
                    clearInterval(interval);
                    resolve(s);
                } else if (s.status === 'error') {
                    clearInterval(interval);
                    reject(new Error(s.message));
                }
            } catch (e) {
                clearInterval(interval);
                reject(e);
            }
        }, 1000);
    });
}

// Label Polling
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
        } catch (error) { console.error(error); }
    }, 1000);
}

// Cluster Polling
function startClusterPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            clusterServerLogs.innerHTML = '';
            addLog(clusterServerLogs, 'info', `Status: ${status.status}`);
            status.details.forEach(detail => {
                let type = 'info';
                if (detail.includes('Error')) type = 'error';
                else if (detail.includes('Complete')) type = 'success';
                const div = document.createElement('div');
                div.className = `log-entry ${type}`;
                div.textContent = detail;
                clusterServerLogs.appendChild(div);
            });
            clusterServerLogs.scrollTop = clusterServerLogs.scrollHeight;

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                clusterBtn.disabled = false;
                clusterBtn.textContent = 'Run Clustering';
                addLog(clusterServerLogs, 'success', 'Clustering Finished!');
                clusterOutputPath.textContent = status.output_file;

                // Show params update
                clusterResultsList.innerHTML = `
                    <li><strong>Min Samples (n):</strong> ${paramN.value}</li>
                    <li><strong>Neighbors (k):</strong> ${paramK.value}</li>
                    <li><strong>Output:</strong> ${status.output_file}</li>
                `;

                // Render K-Distance Graph if available
                if (status.k_distances && status.optimal_eps) {
                    renderElbowChart(status.k_distances, status.optimal_eps);
                    elbowChartCard.classList.remove('hidden');
                }

                loadPreview(clusterPreviewTable, clusterPreviewCard, true);
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                clusterBtn.disabled = false;
                clusterBtn.textContent = 'Retry';
                addLog(clusterServerLogs, 'error', 'Stopped due to error.');
            }
        } catch (error) { console.error(error); }
    }, 1000);
}

// Volatility Polling
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
                volParamsList.innerHTML = `<li><strong>Window Size:</strong> ${volWindowInput.value}</li>`;
                loadPreview(volPreviewTable, volPreviewCard, true);
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                computeVolBtn.disabled = false;
                computeVolBtn.textContent = 'Retry';
                addLog(volServerLogs, 'error', 'Stopped due to error.');
            }
        } catch (error) { console.error(error); }
    }, 1000);
}

// Default Polling (Indicators)
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


function renderElbowChart(kDistances, optimalEps) {
    // Destroy existing chart if it exists
    if (elbowChart) {
        elbowChart.destroy();
    }

    const ctx = elbowChartCanvas.getContext('2d');

    // Find index where eps is closest to optimal
    let elbowIndex = 0;
    let minDiff = Math.abs(kDistances[0] - optimalEps);
    for (let i = 0; i < kDistances.length; i++) {
        const diff = Math.abs(kDistances[i] - optimalEps);
        if (diff < minDiff) {
            minDiff = diff;
            elbowIndex = i;
        }
    }

    elbowChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: kDistances.length }, (_, i) => i),
            datasets: [
                {
                    label: 'K-Distance',
                    data: kDistances,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: true
                },
                {
                    label: `Elbow Point (ε=${optimalEps.toFixed(4)})`,
                    data: [{ x: elbowIndex, y: optimalEps }],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 1)',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'K-Distance Graph - Elbow Method for Epsilon Selection',
                    font: { size: 14 }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        title: (items) => `Point: ${items[0].label}`,
                        label: (item) => `Distance: ${item.parsed.y.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Points (sorted by k-distance)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'K-Distance'
                    },
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

function copyPath(elementId) {
    // If no ID provided, fallback to default (old behavior)
    const el = document.getElementById(elementId) || outputPathDisplay;
    const path = el.textContent;
    navigator.clipboard.writeText(path).then(() => {
        alert('Path copied to clipboard!');
    });
}


// ===================== ML TRAINING TAB =====================

// ML Training Elements
const mlAlgorithm = document.getElementById('mlAlgorithm');
const mlValidation = document.getElementById('mlValidation');
const mlSegmentMode = document.getElementById('mlSegmentMode');
const mlInputPath = document.getElementById('mlInputPath');
const mlCsvFile = document.getElementById('mlCsvFile');
const mlWindowSize = document.getElementById('mlWindowSize');
const mlWindowLabel = document.getElementById('mlWindowLabel');
const mlWindowHint = document.getElementById('mlWindowHint');
const mlValidationDesc = document.getElementById('mlValidationDesc');
const mlTrainBtn = document.getElementById('mlTrainBtn');
const mlServerLogs = document.getElementById('mlServerLogs');
const mlMetricsList = document.getElementById('mlMetricsList');
const mlClassificationCard = document.getElementById('mlClassificationCard');
const mlClassMetricsList = document.getElementById('mlClassMetricsList');
const mlConfusionBtn = document.getElementById('mlConfusionBtn');
const mlFoldsCard = document.getElementById('mlFoldsCard');
const mlFoldsTable = document.getElementById('mlFoldsTable');
const mlFilesCard = document.getElementById('mlFilesCard');
const mlFilesList = document.getElementById('mlFilesList');
const cmModal = document.getElementById('cmModal');
const cmModalClose = document.getElementById('cmModalClose');
const cmBody = document.getElementById('cmBody');
const cmLegend = document.getElementById('cmLegend');

let selectedMlFile = null;

// Validation Strategy & Segment Mode Switcher
function updateMlHints() {
    const val = mlValidation.value;
    const mode = mlSegmentMode.value;
    const unit = mode === 'month' ? 'months' : 'segments';

    if (val === 'rolling') {
        mlWindowLabel.textContent = `Window Size (${unit})`;
        mlWindowHint.textContent = `Train on N ${unit}, test on next`;
        mlValidationDesc.value = `Rolling Window: Train on a fixed-size window (e.g., ${unit} 1–6), test on ${unit} 7.\nThen roll forward: train on ${unit} 2–7, test on ${unit} 8, and so on.\nCollects performance metrics across all test windows.`;
    } else {
        mlWindowLabel.textContent = `Initial Window (${unit})`;
        mlWindowHint.textContent = `Start with N ${unit}, expand each fold`;
        mlValidationDesc.value = `Walk-Forward: Train on an expanding window (start with N ${unit}).\nTest on the next ${unit}. Add that ${unit} to training, retrain, test on the next, and so on.\nCollects performance metrics of prediction across all expanding windows.`;
    }
}

mlValidation.addEventListener('change', updateMlHints);
mlSegmentMode.addEventListener('change', updateMlHints);

// ML File Browse
mlCsvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedMlFile = file;
        mlInputPath.value = `[Upload] ${file.name}`;
    }
});

// Tab switching: auto-fill ML input from previous step output
document.querySelectorAll('.nav-links li').forEach(li => {
    li.addEventListener('click', () => {
        const tabId = li.getAttribute('data-tab');
        if (tabId === 'mltraining') {
            // Auto-fill from Volatility/Label output
            if (volOutputPath && volOutputPath.textContent !== '...' && !mlInputPath.value) {
                mlInputPath.value = volOutputPath.textContent;
            }
        }
    });
});

// ML Train Button
mlTrainBtn.addEventListener('click', async () => {
    let filePath = mlInputPath.value;
    const algorithm = mlAlgorithm.value;
    const validationType = mlValidation.value;
    const segmentMode = mlSegmentMode.value;
    const windowSize = mlWindowSize.value;

    // Handle file upload if user browsed a file
    if (selectedMlFile) {
        addLog(mlServerLogs, 'info', 'Uploading file...');
        const formData = new FormData();
        formData.append('file', selectedMlFile);
        try {
            const upResp = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            if (!upResp.ok) throw new Error('Upload failed');
            addLog(mlServerLogs, 'info', 'Processing base indicators...');
            await pollGenericForCompletion();
            const s = await (await fetch(`${API_URL}/status`)).json();
            filePath = s.output_file;
            selectedMlFile = null;
        } catch (e) {
            addLog(mlServerLogs, 'error', 'Upload failed: ' + e.message);
            return;
        }
    } else if (!filePath || filePath === '...' || filePath === '') {
        // Fallback to last output
        if (volOutputPath && volOutputPath.textContent !== '...') {
            filePath = volOutputPath.textContent;
        } else {
            addLog(mlServerLogs, 'error', 'No input file specified. Please browse or paste a file path.');
            return;
        }
    }

    if (filePath.startsWith('[Upload]')) {
        addLog(mlServerLogs, 'error', 'File upload did not complete. Please retry.');
        return;
    }

    // Reset UI
    mlTrainBtn.disabled = true;
    mlTrainBtn.textContent = 'Training...';
    mlClassificationCard.classList.add('hidden');
    mlFoldsCard.classList.add('hidden');
    mlFilesCard.classList.add('hidden');
    mlServerLogs.innerHTML = '';
    mlMetricsList.innerHTML = '<li>Training in progress...</li>';

    const validationLabel = validationType === 'rolling' ? 'Rolling Window' : 'Walk-Forward';
    addLog(mlServerLogs, 'system', `Starting ${algorithm} with ${validationLabel} (window=${windowSize}) on ${filePath}...`);

    try {
        const url = new URL(`${API_URL}/train-ml`);
        url.searchParams.append('file_path', filePath);
        url.searchParams.append('algorithm', algorithm);
        url.searchParams.append('validation_type', validationType);
        url.searchParams.append('window_size', windowSize);
        url.searchParams.append('segment_mode', segmentMode);

        const resp = await fetch(url, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Request failed');
        }

        startMlPolling();

    } catch (error) {
        addLog(mlServerLogs, 'error', `Error: ${error.message}`);
        mlTrainBtn.disabled = false;
        mlTrainBtn.textContent = 'Start Training';
    }
});


// ML Polling
function startMlPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            mlServerLogs.innerHTML = '';
            addLog(mlServerLogs, 'info', `Status: ${status.status}`);
            status.details.forEach(detail => {
                let type = 'info';
                if (detail.includes('Error') || detail.includes('error')) type = 'error';
                else if (detail.includes('Complete') || detail.includes('Done')) type = 'success';
                else if (detail.includes('Fold')) type = 'info';
                const div = document.createElement('div');
                div.className = `log-entry ${type}`;
                div.textContent = detail;
                mlServerLogs.appendChild(div);
            });
            mlServerLogs.scrollTop = mlServerLogs.scrollHeight;

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                mlTrainBtn.disabled = false;
                mlTrainBtn.textContent = 'Start Training';
                addLog(mlServerLogs, 'success', 'Training Finished!');

                // Load full results
                loadMlResults();

            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                mlTrainBtn.disabled = false;
                mlTrainBtn.textContent = 'Retry';
                addLog(mlServerLogs, 'error', 'Stopped due to error.');
            }
        } catch (error) { console.error(error); }
    }, 1000);
}


// Load ML Results from API
async function loadMlResults() {
    try {
        const resp = await fetch(`${API_URL}/ml-results`);
        const data = await resp.json();

        if (data.error) {
            addLog(mlServerLogs, 'error', data.error);
            return;
        }

        // Regression Metrics
        const reg = data.regression_metrics || {};
        const params = data.parameters || {};
        const paramsHtml = Object.entries(params).map(([k, v]) => `<li><strong>${k}:</strong> ${v}</li>`).join('');

        mlMetricsList.innerHTML = `
            <li><strong>Model:</strong> ${data.model || 'N/A'}</li>
            <li><strong>Validation:</strong> ${data.validation || 'N/A'}</li>
            <li><strong>Segmentation:</strong> ${data.segment_mode || 'N/A'}</li>
            <li style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;"><strong>Parameters:</strong></li>
            ${paramsHtml}
            <li style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;"><strong>Overall Metrics:</strong></li>
            <li><strong>MSE:</strong> ${(reg.overall_mse || 0).toFixed(6)}</li>
            <li><strong>MAE:</strong> ${(reg.overall_mae || 0).toFixed(6)}</li>
            <li><strong>RMSE:</strong> ${(reg.overall_rmse || 0).toFixed(6)}</li>
            <li><strong>R²:</strong> <span class="${reg.overall_r2 > 0.5 ? 'metric-good' : reg.overall_r2 > 0 ? 'metric-warn' : 'metric-bad'}">${(reg.overall_r2 || 0).toFixed(4)}</span></li>
        `;

        // Classification Metrics
        const cls = data.classification_metrics || {};
        if (cls.accuracy !== undefined) {
            mlClassificationCard.classList.remove('hidden');
            mlClassMetricsList.innerHTML = `
                <li><strong>Accuracy:</strong> <span class="${cls.accuracy > 0.7 ? 'metric-good' : cls.accuracy > 0.5 ? 'metric-warn' : 'metric-bad'}">${(cls.accuracy * 100).toFixed(2)}%</span></li>
                <li><strong>Precision (macro):</strong> ${(cls.precision_macro || 0).toFixed(4)}</li>
                <li><strong>Recall (macro):</strong> ${(cls.recall_macro || 0).toFixed(4)}</li>
                <li><strong>F1 Score (macro):</strong> ${(cls.f1_macro || 0).toFixed(4)}</li>
            `;

            if (data.thresholds) {
                const t = data.thresholds;
                mlClassMetricsList.innerHTML += `
                    <li style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;">
                        <strong>Label Thresholds:</strong><br>
                        Low ≤ ${(t.low_max || 0).toFixed(6)}<br>
                        Medium: ${(t.med_min || 0).toFixed(6)} – ${(t.med_max || 0).toFixed(6)}<br>
                        High ≥ ${(t.high_min || 0).toFixed(6)}
                    </li>
                `;
            }
        }

        // Fold Results Table
        const folds = data.folds || [];
        if (folds.length > 0) {
            mlFoldsCard.classList.remove('hidden');
            const thead = mlFoldsTable.querySelector('thead');
            const tbody = mlFoldsTable.querySelector('tbody');
            thead.innerHTML = '<tr><th>Fold</th><th>Train Rows</th><th>Test Rows</th><th>MSE</th><th>MAE</th><th>R²</th></tr>';
            tbody.innerHTML = '';
            folds.forEach(f => {
                const r2Class = f.r2 > 0.5 ? 'metric-good' : f.r2 > 0 ? 'metric-warn' : 'metric-bad';
                tbody.innerHTML += `
                    <tr>
                        <td>${f.fold}</td>
                        <td>${f.train_rows}</td>
                        <td>${f.test_rows}</td>
                        <td>${f.mse.toFixed(6)}</td>
                        <td>${f.mae.toFixed(6)}</td>
                        <td class="${r2Class}">${f.r2.toFixed(4)}</td>
                    </tr>
                `;
            });
        }

        // Saved Files
        const files = data.saved_files || {};
        if (Object.keys(files).length > 0) {
            mlFilesCard.classList.remove('hidden');
            mlFilesList.innerHTML = '';
            for (const [key, path] of Object.entries(files)) {
                mlFilesList.innerHTML += `<li><strong>${key}:</strong> <span class="highlight-path">${path}</span></li>`;
            }
        }

    } catch (error) {
        console.error('Failed to load ML results:', error);
        addLog(mlServerLogs, 'error', 'Failed to load results.');
    }
}


// Confusion Matrix Modal
mlConfusionBtn.addEventListener('click', async () => {
    try {
        const resp = await fetch(`${API_URL}/ml-confusion-matrix`);
        const data = await resp.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        renderConfusionMatrix(data);
        cmModal.classList.remove('hidden');

    } catch (error) {
        alert('Failed to load confusion matrix: ' + error.message);
    }
});

cmModalClose.addEventListener('click', () => {
    cmModal.classList.add('hidden');
});

// Close modal on overlay click
cmModal.addEventListener('click', (e) => {
    if (e.target === cmModal) {
        cmModal.classList.add('hidden');
    }
});

function renderConfusionMatrix(data) {
    const { matrix, labels } = data;
    if (!matrix || !labels) {
        cmBody.innerHTML = '<p style="color: var(--text-muted);">No data available.</p>';
        return;
    }

    // Find max value for heatmap scaling
    let maxVal = 0;
    matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

    // Color function: green diagonal (correct), red off-diagonal (errors)
    function getCellColor(val, isCorrect) {
        const intensity = maxVal > 0 ? val / maxVal : 0;
        if (isCorrect) {
            // Green tones for correct predictions
            const r = Math.round(20 + (35 - 20) * (1 - intensity));
            const g = Math.round(60 + (134 - 60) * intensity);
            const b = Math.round(30 + (54 - 30) * (1 - intensity));
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Red tones for misclassifications
            const r = Math.round(40 + (218 - 40) * intensity);
            const g = Math.round(30 + (54 - 30) * (1 - intensity));
            const b = Math.round(30 + (51 - 30) * (1 - intensity));
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    let html = '<table class="cm-table">';

    // Header row
    html += '<tr><th></th>';
    labels.forEach(label => {
        html += `<th>${label}<br><small>(Predicted)</small></th>`;
    });
    html += '</tr>';

    // Data rows
    matrix.forEach((row, i) => {
        html += `<tr><td class="header-label">${labels[i]}<br><small>(Actual)</small></td>`;
        row.forEach((val, j) => {
            const isCorrect = i === j;
            const bgColor = getCellColor(val, isCorrect);
            html += `<td style="background: ${bgColor}; color: white;">${val}</td>`;
        });
        html += '</tr>';
    });

    html += '</table>';
    cmBody.innerHTML = html;

    // Legend
    cmLegend.innerHTML = `
        <div class="cm-legend-item">
            <div class="cm-legend-swatch" style="background: rgb(35, 134, 54);"></div>
            <span>Correct (diagonal)</span>
        </div>
        <div class="cm-legend-item">
            <div class="cm-legend-swatch" style="background: rgb(218, 54, 51);"></div>
            <span>Misclassified (off-diagonal)</span>
        </div>
        <div class="cm-legend-item">
            <span style="font-style: italic;">Rows = Actual, Columns = Predicted</span>
        </div>
    `;
}


// ===================== HYBRID-ML TAB =====================

// Hybrid Elements
const hybridClusterSource = document.getElementById('hybridClusterSource');
const hybridFilePath = document.getElementById('hybridFilePath');
const hybridCsvFile = document.getElementById('hybridCsvFile');
const hybridFileHint = document.getElementById('hybridFileHint');
const hybridAlgorithm = document.getElementById('hybridAlgorithm');
const hybridValidation = document.getElementById('hybridValidation');
const hybridSegmentMode = document.getElementById('hybridSegmentMode');
const hybridWindowSize = document.getElementById('hybridWindowSize');
const hybridWindowLabel = document.getElementById('hybridWindowLabel');
const hybridWindowHint = document.getElementById('hybridWindowHint');
const hybridTrainBtn = document.getElementById('hybridTrainBtn');
const hybridServerLogs = document.getElementById('hybridServerLogs');
const hybridMetricsList = document.getElementById('hybridMetricsList');
const hybridClassificationCard = document.getElementById('hybridClassificationCard');
const hybridClassMetricsList = document.getElementById('hybridClassMetricsList');
const hybridConfusionBtn = document.getElementById('hybridConfusionBtn');
const hybridFoldsCard = document.getElementById('hybridFoldsCard');
const hybridFoldsTable = document.getElementById('hybridFoldsTable');
const hybridFilesCard = document.getElementById('hybridFilesCard');
const hybridFilesList = document.getElementById('hybridFilesList');

let selectedHybridFile = null;

// Fetch cluster files from backend and populate
async function fetchClusterFiles() {
    try {
        const resp = await fetch(`${API_URL}/cluster-files`);
        const data = await resp.json();
        return data;
    } catch (e) {
        console.error('Failed to fetch cluster files:', e);
        return { dbscan: '', kmeans: '' };
    }
}

// Cluster Source Switcher — auto-populate file path
hybridClusterSource.addEventListener('change', async () => {
    selectedHybridFile = null;
    const source = hybridClusterSource.value;
    const files = await fetchClusterFiles();
    const path = files[source] || '';

    if (path) {
        hybridFilePath.value = path;
        hybridFileHint.textContent = `Using ${source.toUpperCase()} output file`;
        hybridFileHint.style.color = 'var(--success)';
    } else {
        hybridFilePath.value = '';
        hybridFileHint.textContent = `No ${source.toUpperCase()} file found. Run clustering first or browse manually.`;
        hybridFileHint.style.color = 'var(--error)';
    }
});

// Auto-populate on tab switch
document.querySelectorAll('.nav-links li').forEach(li => {
    li.addEventListener('click', async () => {
        if (li.getAttribute('data-tab') === 'hybridml' && !hybridFilePath.value) {
            hybridClusterSource.dispatchEvent(new Event('change'));
        }
    });
});

// Hybrid File Browse (override)
hybridCsvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedHybridFile = file;
        hybridFilePath.value = `[Upload] ${file.name}`;
        hybridFileHint.textContent = `Manual file selected: ${file.name}`;
        hybridFileHint.style.color = 'var(--text-muted)';
    }
});

// Validation Strategy & Segment Mode Switcher
function updateHybridHints() {
    const val = hybridValidation.value;
    const mode = hybridSegmentMode.value;
    const unit = mode === 'month' ? 'months' : 'segments';

    if (val === 'rolling') {
        hybridWindowLabel.textContent = `Window Size (${unit})`;
        hybridWindowHint.textContent = `Train on N ${unit}, test on next`;
    } else {
        hybridWindowLabel.textContent = `Initial Window (${unit})`;
        hybridWindowHint.textContent = `Start with N ${unit}, expand each fold`;
    }
}

hybridValidation.addEventListener('change', updateHybridHints);
hybridSegmentMode.addEventListener('change', updateHybridHints);

// Hybrid Train Button
hybridTrainBtn.addEventListener('click', async () => {
    let filePath = hybridFilePath.value;
    const algorithm = hybridAlgorithm.value;
    const validationType = hybridValidation.value;
    const segmentMode = hybridSegmentMode.value;
    const windowSize = hybridWindowSize.value;
    const clusterSource = hybridClusterSource.value;

    // Handle manual file upload
    if (selectedHybridFile) {
        addLog(hybridServerLogs, 'info', 'Uploading file...');
        const formData = new FormData();
        formData.append('file', selectedHybridFile);
        try {
            const upResp = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
            if (!upResp.ok) throw new Error('Upload failed');
            addLog(hybridServerLogs, 'info', 'Processing...');
            await pollGenericForCompletion();
            const s = await (await fetch(`${API_URL}/status`)).json();
            filePath = s.output_file;
            selectedHybridFile = null;
        } catch (e) {
            addLog(hybridServerLogs, 'error', 'Upload failed: ' + e.message);
            return;
        }
    }

    if (!filePath || filePath === '' || filePath.startsWith('[Upload]')) {
        addLog(hybridServerLogs, 'error', 'No cluster file available. Run clustering first or browse a file.');
        return;
    }

    // Reset UI
    hybridTrainBtn.disabled = true;
    hybridTrainBtn.textContent = 'Training...';
    hybridClassificationCard.classList.add('hidden');
    hybridFoldsCard.classList.add('hidden');
    hybridFilesCard.classList.add('hidden');
    hybridServerLogs.innerHTML = '';
    hybridMetricsList.innerHTML = '<li>Training in progress...</li>';

    const validationLabel = validationType === 'rolling' ? 'Rolling Window' : 'Walk-Forward';
    addLog(hybridServerLogs, 'system', `Starting Hybrid-ML (${clusterSource.toUpperCase()}) with ${validationLabel} (window=${windowSize})...`);
    addLog(hybridServerLogs, 'info', `File: ${filePath}`);

    try {
        const url = new URL(`${API_URL}/train-hybrid`);
        url.searchParams.append('file_path', filePath);
        url.searchParams.append('algorithm', algorithm);
        url.searchParams.append('validation_type', validationType);
        url.searchParams.append('window_size', windowSize);
        url.searchParams.append('segment_mode', segmentMode);
        url.searchParams.append('cluster_source', clusterSource);

        const resp = await fetch(url, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Request failed');
        }

        startHybridPolling();

    } catch (error) {
        addLog(hybridServerLogs, 'error', `Error: ${error.message}`);
        hybridTrainBtn.disabled = false;
        hybridTrainBtn.textContent = 'Start Hybrid Training';
    }
});


// Hybrid Polling
function startHybridPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            hybridServerLogs.innerHTML = '';
            addLog(hybridServerLogs, 'info', `Status: ${status.status}`);
            status.details.forEach(detail => {
                let type = 'info';
                if (detail.includes('Error') || detail.includes('error')) type = 'error';
                else if (detail.includes('Complete') || detail.includes('Done')) type = 'success';
                else if (detail.includes('Fold')) type = 'info';
                const div = document.createElement('div');
                div.className = `log-entry ${type}`;
                div.textContent = detail;
                hybridServerLogs.appendChild(div);
            });
            hybridServerLogs.scrollTop = hybridServerLogs.scrollHeight;

            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                hybridTrainBtn.disabled = false;
                hybridTrainBtn.textContent = 'Start Hybrid Training';
                addLog(hybridServerLogs, 'success', 'Hybrid Training Finished!');
                loadHybridResults();

            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                hybridTrainBtn.disabled = false;
                hybridTrainBtn.textContent = 'Retry';
                addLog(hybridServerLogs, 'error', 'Stopped due to error.');
            }
        } catch (error) { console.error(error); }
    }, 1000);
}


// Load Hybrid Results from API
async function loadHybridResults() {
    try {
        const resp = await fetch(`${API_URL}/hybrid-results`);
        const data = await resp.json();

        if (data.error) {
            addLog(hybridServerLogs, 'error', data.error);
            return;
        }

        // Regression Metrics
        const reg = data.regression_metrics || {};
        const params = data.parameters || {};
        const paramsHtml = Object.entries(params).map(([k, v]) => `<li><strong>${k}:</strong> ${v}</li>`).join('');

        hybridMetricsList.innerHTML = `
            <li><strong>Model:</strong> ${data.model || 'N/A'}</li>
            <li><strong>Validation:</strong> ${data.validation || 'N/A'}</li>
            <li><strong>Segmentation:</strong> ${data.segment_mode || 'N/A'}</li>
            <li><strong>Cluster Source:</strong> ${(data.cluster_source || '').toUpperCase()}</li>
            <li style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;"><strong>Parameters:</strong></li>
            ${paramsHtml}
            <li style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;"><strong>Overall Metrics:</strong></li>
            <li><strong>MSE:</strong> ${(reg.overall_mse || 0).toFixed(6)}</li>
            <li><strong>MAE:</strong> ${(reg.overall_mae || 0).toFixed(6)}</li>
            <li><strong>RMSE:</strong> ${(reg.overall_rmse || 0).toFixed(6)}</li>
            <li><strong>R²:</strong> <span class="${reg.overall_r2 > 0.5 ? 'metric-good' : reg.overall_r2 > 0 ? 'metric-warn' : 'metric-bad'}">${(reg.overall_r2 || 0).toFixed(4)}</span></li>
        `;

        // Classification Metrics
        const cls = data.classification_metrics || {};
        if (cls.accuracy !== undefined) {
            hybridClassificationCard.classList.remove('hidden');
            hybridClassMetricsList.innerHTML = `
                <li><strong>Accuracy:</strong> <span class="${cls.accuracy > 0.7 ? 'metric-good' : cls.accuracy > 0.5 ? 'metric-warn' : 'metric-bad'}">${(cls.accuracy * 100).toFixed(2)}%</span></li>
                <li><strong>Precision (macro):</strong> ${(cls.precision_macro || 0).toFixed(4)}</li>
                <li><strong>Recall (macro):</strong> ${(cls.recall_macro || 0).toFixed(4)}</li>
                <li><strong>F1 Score (macro):</strong> ${(cls.f1_macro || 0).toFixed(4)}</li>
            `;

            if (data.thresholds) {
                const t = data.thresholds;
                hybridClassMetricsList.innerHTML += `
                    <li style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;">
                        <strong>Label Thresholds:</strong><br>
                        Low ≤ ${(t.low_max || 0).toFixed(6)}<br>
                        Medium: ${(t.med_min || 0).toFixed(6)} – ${(t.med_max || 0).toFixed(6)}<br>
                        High ≥ ${(t.high_min || 0).toFixed(6)}
                    </li>
                `;
            }
        }

        // Fold Results Table
        const folds = data.folds || [];
        if (folds.length > 0) {
            hybridFoldsCard.classList.remove('hidden');
            const thead = hybridFoldsTable.querySelector('thead');
            const tbody = hybridFoldsTable.querySelector('tbody');
            thead.innerHTML = '<tr><th>Fold</th><th>Train Rows</th><th>Test Rows</th><th>MSE</th><th>MAE</th><th>R²</th></tr>';
            tbody.innerHTML = '';
            folds.forEach(f => {
                const r2Class = f.r2 > 0.5 ? 'metric-good' : f.r2 > 0 ? 'metric-warn' : 'metric-bad';
                tbody.innerHTML += `
                    <tr>
                        <td>${f.fold}</td>
                        <td>${f.train_rows}</td>
                        <td>${f.test_rows}</td>
                        <td>${f.mse.toFixed(6)}</td>
                        <td>${f.mae.toFixed(6)}</td>
                        <td class="${r2Class}">${f.r2.toFixed(4)}</td>
                    </tr>
                `;
            });
        }

        // Saved Files
        const files = data.saved_files || {};
        if (Object.keys(files).length > 0) {
            hybridFilesCard.classList.remove('hidden');
            hybridFilesList.innerHTML = '';
            for (const [key, path] of Object.entries(files)) {
                hybridFilesList.innerHTML += `<li><strong>${key}:</strong> <span class="highlight-path">${path}</span></li>`;
            }
        }

    } catch (error) {
        console.error('Failed to load Hybrid results:', error);
        addLog(hybridServerLogs, 'error', 'Failed to load results.');
    }
}


// Hybrid Confusion Matrix
hybridConfusionBtn.addEventListener('click', async () => {
    try {
        const resp = await fetch(`${API_URL}/hybrid-confusion-matrix`);
        const data = await resp.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        renderConfusionMatrix(data);
        cmModal.classList.remove('hidden');

    } catch (error) {
        alert('Failed to load confusion matrix: ' + error.message);
    }
});
