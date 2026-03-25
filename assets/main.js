// This File is imported into the webview via the script tag
// This file is used to communicate with VS Code
// It uses the vscode namespace to enable communication via postMessage()
// It uses the acquireVsCodeApi() function to get a reference to VS Code

let figureResizeObserver = null;
let isVerticalDivider = null;
let activeDivider = null;
let startPos = 0;
let startSize1 = 0;
let startSize2 = 0;

async function updateHeaderInfo(data) {
    /**
     * This function updated the header info section of the webview
     * It updates the selectable options for the HDU, colormap, and scale.
     * 
     * @param {string} data - JSON stringified object containing the file, selectedHdu, options, and vmin/vmax (clim)
     * 
     * @returns {void} 
     */

    // Set initial section sizes on first document open
    const optionsSection = document.getElementById('optionsSection')
    const hduSelector = document.getElementById('hduSelector')

    if(!optionsSection.style.width || !optionsSection.style.width.includes('px')) {
        initSectionSizes()
    }

    // Parse the data
    const { file, selectedHdu, options } = JSON.parse(data);

    const hduData = file[`hdu${selectedHdu}`];
    const tableObj = hduData?.table;
    const hasTable = !!tableObj && typeof tableObj === 'object' && Object.keys(tableObj).length > 0;

    // -------- Begin HDU selector -------- //
    const hduLabels = options.hduLabels ?? 'index';

    // define function for escaping special HTML characters from HDU names
    // Source - https://stackoverflow.com/a/6234804
    // Posted by bjornd, modified by community. See post 'Timeline' for change history
    // Retrieved 2026-03-21, License - CC BY-SA 4.0
    const escapeHtml = unsafe => {
    return unsafe
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    };

    hduSelector.innerHTML = Object.keys(file).map((element, index) => {
        const isSelected = index === selectedHdu;

        const header = file[element]?.header ?? {};
        const rawName = header.EXTNAME ?? header.HDUNAME ?? null;
        const hduName = (typeof rawName === 'string' && rawName.trim()) ? rawName.trimStart() : null;

        let label = `HDU ${index}`;
        if (hduLabels === 'name' && hduName) {
            label = hduName;
        } else if (hduLabels === 'nameAndIndex' && hduName) {
            label = `${hduName} (HDU ${index})`;
        }

        return `<vscode-option value="${index}" ${isSelected ? 'selected' : ''}>${escapeHtml(label)}</vscode-option>`;
    }).join('');

    // Listen for change event on hduSelector
    hduSelector.addEventListener('change', function() {
        const newSelectedHdu = parseInt(this.value, 10); // Parse to integer
        // Post a message to VS Code
        vscode.postMessage({
            command: 'hduSelectorChanged',
            newSelectedHdu: newSelectedHdu
        });
    });
    // -------- End of HDU selector -------- //

    if (hasTable) {
        tablePlotOptions(data);
    } else {
        imagePlotOptions(data);
    }

    // Get a reference to VS Code
    const headerInfoElement = document.getElementById('headerInfo');
    const searchInput = document.getElementById('input');

    // Add an event listener to the search input
    searchInput.addEventListener('input', function() {
        handleSearch(searchInput);
    });

    // -------- Begin header info table -------- //
    // Display header keys and values in a table for the selected HDU
    const selectedHeader = file[`hdu${selectedHdu}`].header ?? {};
    const tableHtml = `
        <vscode-data-grid>
            <vscode-data-grid-row row-type="sticky-header">
                <vscode-data-grid-cell cell-type="columnheader" grid-column="1">Key</vscode-data-grid-cell>
                <vscode-data-grid-cell cell-type="columnheader" grid-column="2">Value</vscode-data-grid-cell>
            </vscode-data-grid-row>
            ${Object.entries(selectedHeader)
                .filter(([key]) => key !== 'END')
                .map(([key, value]) => `
                <vscode-data-grid-row>
                    <vscode-data-grid-cell grid-column="1" style="font-weight: semi-bold;">${key}</vscode-data-grid-cell>
                    <vscode-data-grid-cell grid-column="2">${value}</vscode-data-grid-cell>
                </vscode-data-grid-row>
            `).join('')}
        </vscode-data-grid>
    `;
    headerInfoElement.innerHTML = tableHtml;
    // -------- End of header info table -------- //

    // Display the image if it exists, otherwise display placeholder text
    if (file[`hdu${selectedHdu}`].encoded_image) {
        document.getElementById('imageTableDiv').style.display = 'flex';
        document.getElementById('placeholderText').style.display = 'none';
        document.getElementById('imageErrorText').style.display = 'none';
    } else {
        document.getElementById('imageTableDiv').style.display = 'none';
        document.getElementById('placeholderText').style.display = 'block';
        document.getElementById('imageErrorText').style.display = 'none';
    }

    return;
}

async function imagePlotOptions(data) {
    
    // Parse the data
    const { file, selectedHdu, options } = JSON.parse(data);

    // Get selector elements
    const selector1 = document.getElementById('selector1');
    const selector2 = document.getElementById('selector2');
    const selector3 = document.getElementById('selector3');

    // -------- Begin colormap selector -------- //
    const colormapOptions = ['viridis', 'gray', 'rainbow', 'hot', 'jet', 'aips0'].map((color) => {
        const isSelected = color === options.colormap;
        return `<vscode-radio value="${color}" ${isSelected ? 'checked' : ''} onclick="updateColor('${color}')">${color}</vscode-radio>`;
    }).join('');

    selector1.innerHTML = `<vscode-radio-group id="colorSelector">
    <label slot="label">Colormap</label>${colormapOptions}
    </vscode-radio-group>`;

    // -------- End of colormap selector -------- //

    const capitalizeFirstLetter = (string) => {
        // Capitalize the first letter of a string
        return string.charAt(0).toUpperCase() + string.slice(1);
    };
        
    window.addEventListener('resize', adjustScale);

    // -------- Begin scale selector -------- //
    const scaleOptions = ['linear', 'logarithmic', 'power', 'sqrt',  'squared'].map((value) => {
        const label = capitalizeFirstLetter(value);
        const isSelected = value === options.scale;
        return `<vscode-radio value="${value}" ${isSelected ? 'checked' : ''} onclick="updateScale('${value}')">${label}</vscode-radio>`;
    }).join('');
    selector2.innerHTML = `<vscode-radio-group id="scaleSelector">
    <label slot="label">Scale</label>${scaleOptions}
    </vscode-radio-group>`;
    // -------- End of scale selector -------- //

    // -------- Begin clim input -------- //
    const selectedHduData = file[`hdu${selectedHdu}`] ?? {};
    const renderedClim = Array.isArray(selectedHduData.clim) ? selectedHduData.clim : [null, null];
    const values = { "vmin" : renderedClim[0], "vmax": renderedClim[1] };
    const climSettings = ["vmin", "vmax"].map((label) => {
        return `<vscode-text-field type="text" id="${label}" label="${label}" 
        value="${formatClimValue(values?.[label])}" onchange="updateClim()">${label}</vscode-text-field>`;
    }).join('');

    const resetButton = `<vscode-button appearance="secondary" onclick="resetClim()">Reset</vscode-button>`;

    selector3.innerHTML = `<div id="climInput">${climSettings + resetButton}</div>`;
    // -------- End of clim input -------- //
}

async function tablePlotOptions(data) {
    /**
     * Updates the options section for table/scatter plot visualization
     * Replaces colormap/scale/clim with x_col, y_col, scales, and point size
     * 
     * @param {string} data - JSON stringified object containing file, selectedHdu, options
     */
    
    const { file, selectedHdu, options } = JSON.parse(data);
    const hduData = file[`hdu${selectedHdu}`];
    const tableObj = hduData?.table;
    
    if (!tableObj || typeof tableObj !== 'object') {
        return; // No table data, skip
    }
    
    const colNames = Object.keys(tableObj) ?? [];
    
    // Get selector elements
    const selector1 = document.getElementById('selector1');
    const selector2 = document.getElementById('selector2');
    const selector3 = document.getElementById('selector3');

    // -------- Begin column selectors -------- //
    const xColOptions = colNames.map(col => {
        const isSelected = col === (options.cols?.x ?? colNames[0]);
        return `<vscode-option value="${col}" ${isSelected ? 'selected' : ''}>${col}</vscode-option>`;
    }).join('');
    
    const yColOptions = colNames.map(col => {
        const isSelected = col === (options.cols?.y ?? colNames[1] ?? colNames[0]);
        return `<vscode-option value="${col}" ${isSelected ? 'selected' : ''}>${col}</vscode-option>`;
    }).join('');
    
    selector1.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <label slot="label">X Column</label>
            <vscode-dropdown id="xColSelect" onchange="updateCols()">
                ${xColOptions}
            </vscode-dropdown>
            <label slot="label">Y Column</label>
            <vscode-dropdown id="yColSelect" onchange="updateCols()">
                ${yColOptions}
            </vscode-dropdown>
        </div>
    `;
    // -------- End of column selectors -------- //
    
    // -------- Begin scale selector -------- //
    const scaleOptions = ['linear', 'log'];
    const xScaleRadios = scaleOptions.map(value => {
        const label = value.charAt(0).toUpperCase() + value.slice(1);
        const isSelected = value === (options.tableScale.x);
        return `<vscode-radio value="${value}" ${isSelected ? 'checked' : ''} onclick="updateXTableScale(this.value)">${label}</vscode-radio>`;
    }).join('');
    
    const yScaleRadios = scaleOptions.map(value => {
        const label = value.charAt(0).toUpperCase() + value.slice(1);
        const isSelected = value === (options.tableScale.y);
        return `<vscode-radio value="${value}" ${isSelected ? 'checked' : ''} onclick="updateYTableScale(this.value)">${label}</vscode-radio>`;
    }).join('');
    
    selector2.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px; margin: 2px">
            <label slot="label">X Scale</label>
            <vscode-radio-group>
                ${xScaleRadios}
            </vscode-radio-group>
        </div>
        <div style="display: flex; align-items: center; gap: 12px; margin: 2px">
            <label slot="label">Y Scale</label>
            <vscode-radio-group>
                ${yScaleRadios}
            </vscode-radio-group>
        </div>
    `;
    // -------- End of scale selector -------- //
    
    // -------- Begin point size and plot type selector -------- //
    const pointSize = options.pointSize ?? 5;
    const plotTypes = ['line', 'scatter', 'line_scatter', 'step'];
    const plotTypeOptions = plotTypes.map(type => {
        const label = type.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        const isSelected = type === (options.plotType ?? 'line');
        return `<vscode-radio value="${type}" ${isSelected ? 'checked' : ''} onclick="updatePlotType('${type}')">  ${label}</vscode-radio>`;
    }).join('');
    
    selector3.innerHTML = `
        <label for="pointSizeSlider">Point Size</label>
        <input type="range" id="pointSizeSlider" min="1" max="20" value="${pointSize}" 
            style="width: 150px; margin: 0 10px;" oninput="interactivePointSize(this.value)" onchange="updatePointSize(this.value)">
        <span id="pointSizeValue">${pointSize}</span>
        <div style="margin-top: 12px;">
            <vscode-radio-group>
                <label slot="label">Plot Type</label>
                ${plotTypeOptions}
            </vscode-radio-group>
        </div>
    `;
    // -------- End of point size and plot type selector -------- //
}

function attachFigureResizeObserver() {
    const figure = document.querySelector('#imageDiv .mpld3-figure');
    if (!figure) return;

    if (figureResizeObserver) {
        figureResizeObserver.disconnect();
    }

    figureResizeObserver = new ResizeObserver(() => {
        adjustScale()
    })

    figureResizeObserver.observe(figure);
    requestAnimationFrame(() => adjustScale());
}

function adjustScale() {
    // Adjust the scale of the image to fit the webview
    // Get the image element
    var imageDiv = document.getElementById('mpld3Figure2');
    var imageSection = document.getElementById('imageSection');
    var mpld3Figure = document.querySelector('#imageDiv .mpld3-figure');

    if (mpld3Figure !== null) {
        var bbox = mpld3Figure.getBBox();
        var aspectRatioWidth = (imageSection.clientWidth - 20) / bbox.width ;
        var aspectRatioHeight = (imageSection.clientHeight - 40) / bbox.height ;
        var aspectRatio = Math.min(aspectRatioWidth, aspectRatioHeight);

        // Apply the scaling transformation
        imageDiv.style.transform = 'scale(' + aspectRatio + ')';

        var svgElement = document.querySelector('.mpld3-toolbar');

        // Change the y property to a new value, for example, 200
        svgElement.setAttribute('y', '40');
        svgElement.setAttribute('x', 10*bbox.width / bbox.height);
    }
}

// Listen for messages from the webview
window.addEventListener('message', async event => {
    const message = event.data;
    
    if (message.command === 'updateHeaderInfo') {
        await updateHeaderInfo(message.data);
        attachFigureResizeObserver();
    }

    if (message.command === 'updateImage') {
        document.getElementById('imageErrorText').style.display = 'none';
        document.getElementById('imageDiv').style.display = 'flex';
        await updateImageContent(message.data);
        setLoading(false);
        attachFigureResizeObserver();
    }

    if (message.command === 'updateTheme') {
        await updateTheme(message.data);
        attachFigureResizeObserver();
    }

    if (message.command === 'renderError') {
        setLoading(false);
        console.log(`Error encountered while rerendering for ${message.reason}.`);
        console.log(message.error);
        document.getElementById('imageTableDiv').style.display = 'none';
        document.getElementById('placeholderText').style.display = 'none';
        document.getElementById('imageErrorText').style.display = 'block';
    }
});

// Define the updateImageContent function
async function updateImageContent(data) {
    /**
     * This function updates the image content of the webview
     * 
     * @param {string} data - JSON stringified object containing the file, selectedHdu, and options
     * 
     * @returns {void}
     *  
    */

    // Get a reference to VS Code
    const imageTableDiv = document.getElementById('imageTableDiv');
    const imageDiv = document.getElementById('imageDiv');

    // Parse the data
    const { file, selectedHdu, _ } = JSON.parse(data);

    // Update the image content
    if (imageTableDiv) {
        const hduData = file[`hdu${selectedHdu}`];
        const tableObj = hduData?.table;
        const hasTable = !!tableObj && typeof tableObj === 'object' && Object.keys(tableObj).length > 0;

        if (hasTable) {
            const table = file[`hdu${selectedHdu}`]['table']
            const colNames = Object.keys(table) ?? [];
            const rowKeys = Object.keys(tableObj[colNames[0]] ?? {});
            const rows = rowKeys.map((rk) =>
                Object.fromEntries(colNames.map((col) => [col, tableObj[col]?.[rk] ?? '']))
            );
            
            if (!document.getElementById('tableDiv')) {
                // Only make the table/plot divide once
                imageTableDiv.innerHTML = `
                <!-- Left pane: Plot -->
                <div id="imageDiv" class="responsive-plot withTable">
                    <!-- Plot goes here -->
                    ${hduData.html_plot}
                </div>
                
                <!-- Vertical divider -->
                <div class="section-divider vertical" style="width: 4px; cursor: col-resize;" data-resize="imageDiv|tableDiv"></div>
                
                <!-- Right pane: Table -->
                <div id="tableDiv">
                    <div id="tableDataInfo">
                        <h2>Data</h2>
                        <vscode-text-field placeholder="Search" id="tableSearch">
                            <span slot="start" appearance="icon" class="codicon codicon-search"></span>
                        </vscode-text-field>
                    </div>
                    <div id="tableData">${renderTableAsDataGrid(colNames, rows)}</div>
                </div>
                `

                // Re-attach divider event listener to new divider
                const newDivider = imageTableDiv.querySelector('.section-divider');
                if (newDivider) {
                    newDivider.addEventListener('mousedown', handleDividerMouseDown);
                }

                const newSearch = document.getElementById('tableSearch');
                if (newSearch) {
                    newSearch.addEventListener('input', () => handleTableSearch(newSearch));
                }
            } else {
                document.getElementById('imageDiv').innerHTML = hduData.html_plot;
                document.getElementById('tableData').innerHTML = renderTableAsDataGrid(colNames, rows);
            }
        } else{
            if (document.getElementById('tableDiv')) {
            // Update the image content
            imageTableDiv.innerHTML = `
            <div id="imageDiv" class="responsive-plot" style="width: 100%">
                ${file[`hdu${selectedHdu}`].html_plot}
            </div>
            `
            } else {
                imageDiv.innerHTML = file[`hdu${selectedHdu}`].html_plot;
            }
            updateClimInputFields(data);
        }

        const newScripts = Array.from(imageDiv.querySelectorAll('script'));
        // Re-execute scripts
        newScripts.forEach(script => {
            const newScript = document.createElement('script');
            newScript.textContent = script.textContent;
            newScript.setAttribute('nonce', generateNonce());
            document.body.appendChild(newScript);
        });
    }

    return;
}

function formatClimValue(value) {
    if (!Number.isFinite(value)) return '';
    const abs = Math.abs(value);
    return abs >= 1000 ? value.toExponential(2) : value.toFixed(2);
}

async function updateClimInputFields(data) {
    /**
     * This function updates the input vmin & vmax input fields
     * 
     * @param {string} data - JSON stringified object containing the file, selectedHdu, and options
     * 
     * @returns {void}
     *  
    */

    // Parse the data
    const { file, selectedHdu, _ } = JSON.parse(data);

    // Update input fields
    const selectedHduData = file[`hdu${selectedHdu}`] ?? {};
    const renderedClim = Array.isArray(selectedHduData.clim) ? selectedHduData.clim : [null, null];
    const values = { "vmin" : renderedClim[0], "vmax": renderedClim[1] };

    const vminEl = document.getElementById("vmin")
    const vmaxEl = document.getElementById("vmax")

    if (!vminEl || !vmaxEl) return;

    vminEl.value = formatClimValue(values.vmin);
    vmaxEl.value = formatClimValue(values.vmax);

    return;
}

function generateNonce() {
    // Generate a random nonce
    const randomBytes = new Uint8Array(16);
    crypto.getRandomValues(randomBytes);
    const nonce = Array.from(randomBytes)
      .map(byte => byte.toString(16).padStart(2, '0'))
      .join('');
    return nonce;
  }


// Define the handleSearch function
function handleSearch(searchInput) {
    const searchValue = searchInput.value.toLowerCase();
    const container = document.getElementById('headerInfoSection')

    // Select all rows in the table
    const rows = container.querySelectorAll('vscode-data-grid-row');

    // Filter only the rows that don't have the class .column-header
    const dataRows = Array.from(rows).filter(row => !row.classList.contains('sticky-header'));

    // Loop through each row and check if the key or value contains the search text
    dataRows.forEach(row => {
        const keyCell = row.querySelector('vscode-data-grid-cell[grid-column="1"]');
        const valueCell = row.querySelector('vscode-data-grid-cell[grid-column="2"]');

        if (keyCell.textContent.toLowerCase().includes(searchValue) || valueCell.textContent.toLowerCase().includes(searchValue)) {
            // If the key or value contains the search text, display the row
            row.style.display = '';
        } else {
            // Otherwise, hide the row
            row.style.display = 'none';
        }
    });
}

function handleTableSearch(searchInput) {
    const query = (searchInput?.value ?? '').trim().toLowerCase();
    const container = document.getElementById('tableData');
    if (!container) return;

    // Only data rows (exclude sticky header row)
    const rows = container.querySelectorAll('vscode-data-grid-row:not([row-type="sticky-header"])');

    rows.forEach((row) => {
        const cells = row.querySelectorAll('vscode-data-grid-cell:not([cell-type="columnheader"])');

        // Match if ANY cell value in the row contains the query
        const isMatch =
            query === '' ||
            Array.from(cells).some((cell) =>
                (cell.textContent ?? '').toLowerCase().includes(query)
            );

        row.style.display = isMatch ? '' : 'none';
    });
}

function updateTheme(isDark) {
    document.body.classList.toggle('vscode-dark', !!isDark)
    document.body.classList.toggle('vscode-light', !isDark)
}

function initSectionSizes() {

    // Set initial dimensions on first load (only if not already set)
    const optionsSection = document.getElementById('optionsSection');
    const headerInfoSection = document.getElementById('headerInfoSection');
    const headerInfoContainer = document.getElementById('headerInfoContainer');
    const imageSection = document.getElementById('imageSection');

    optionsSection.style.width = '40%'; // default width
    headerInfoSection.style.width = '60%'; // remaining width
    headerInfoContainer.style.height = '50%'; // default height
    imageSection.style.height = '40%'; // remaining height

    requestAnimationFrame(() => saveSectionSizesInPx())
}

function saveSectionSizesInPx() {
    const optionsSection = document.getElementById('optionsSection');
    const headerInfoSection = document.getElementById('headerInfoSection');
    const headerInfoContainer = document.getElementById('headerInfoContainer');
    const imageSection = document.getElementById('imageSection');

    optionsSection.style.width = optionsSection.getBoundingClientRect().width + 'px';
    headerInfoSection.style.width = headerInfoSection.getBoundingClientRect().width + 'px';
    headerInfoContainer.style.height = headerInfoContainer.getBoundingClientRect().height + 'px';
    imageSection.style.height = imageSection.getBoundingClientRect().height + 'px';

}

// Resizable section by mouse dragging
document.querySelectorAll('.section-divider').forEach(divider => {
    divider.addEventListener('mousedown', handleDividerMouseDown);
});

function handleDividerMouseDown(e) {
    activeDivider = e.target;
    activeDivider.classList.add('dragging');

    const [id1, id2] = activeDivider.dataset.resize.split('|');
    const elem1 = document.getElementById(id1);
    const elem2 = document.getElementById(id2);

    isVerticalDivider = activeDivider.classList.contains('vertical');

    startPos = isVerticalDivider ? e.clientX : e.clientY;
    startSize1 = isVerticalDivider ? elem1.offsetWidth : elem1.offsetHeight;
    startSize2 = isVerticalDivider ? elem2.offsetWidth : elem2.offsetHeight;
    
    document.addEventListener('mousemove', handleDividerMouseMove);
    document.addEventListener('mouseup', handleDividerMouseUp);
}

function handleDividerMouseMove(e) {
    if (!activeDivider) return;
    
    const [id1, id2] = activeDivider.dataset.resize.split('|');
    const elem1 = document.getElementById(id1);
    const elem2 = document.getElementById(id2);

    const cursorPos = isVerticalDivider ? e.clientX : e.clientY;
    const delta = cursorPos - startPos;
    
    if (isVerticalDivider) {
        const newSize1 = startSize1 + delta;
        const newSize2 = startSize2 - delta;
        elem1.style.width = newSize1 + 'px';
        elem2.style.width = newSize2 + 'px';
    } else {
        const newSize1 = startSize1 + delta;
        const newSize2 = startSize2 - delta;
        elem1.style.height = newSize1 + 'px';
        elem2.style.height = newSize2 + 'px';
    }
    attachFigureResizeObserver();
}

function handleDividerMouseUp() {
    if (activeDivider) {
        activeDivider.classList.remove('dragging');
        activeDivider = null;
    }
    document.removeEventListener('mousemove', handleDividerMouseMove);
    document.removeEventListener('mouseup', handleDividerMouseUp);
}

function renderTableAsDataGrid(columns, tableData) {
  const headerRow = `
    <vscode-data-grid-row row-type="sticky-header" >
      ${columns.map((col, i) => `<vscode-data-grid-cell cell-type="columnheader" grid-column="${i + 1}">${col}</vscode-data-grid-cell>`).join('')}
    </vscode-data-grid-row>
  `;
  
  const dataRows = tableData.map(row => `
    <vscode-data-grid-row>
      ${columns.map((col, i) => `<vscode-data-grid-cell grid-column="${i + 1}">${row[col] ?? ''}</vscode-data-grid-cell>`).join('')}
    </vscode-data-grid-row>
  `).join('');
  
  return `<vscode-data-grid>${headerRow}${dataRows}</vscode-data-grid>`;
}