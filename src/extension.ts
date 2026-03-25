// Description: This is the main file of the extension. 
// It contains the code to activate and deactivate the extension, 
// and to register the custom editor provider.

// Import the relevant modules
import * as vscode from 'vscode';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
const osvar = process.platform;
const path = require("path");
const requirements = path.join(__dirname, "python/requirements.txt");
const shellQuote = (s: string) => "'" + s.replace(/'/g,`'"'"'`) + "'";

type Clim = { vmin: number | null; vmax: number | null };
type TableCols = { x: string | null; y: string | null};
type TableScale = { x: 'linear' | 'log' ; y: 'linear' | 'log' };

type RenderOptions = {
  colormap: string;
  scale: 'linear' | 'logarithmic' | 'sqrt' | 'power' | 'squared';
  clim: Clim;
  cols: TableCols;
  tableScale: TableScale;
  pointSize: number;
  plotType: string;
  hduLabels: string;
  saveMode: boolean;
  saveHdu: number;
  savePath: string;
  saveFn: string;
  dpi: string;
};

// Implement the CustomTextEditorProvider interface
class CustomEditorProvider implements vscode.CustomReadonlyEditorProvider {
    private openDocuments: Map<string, CustomDocument> = new Map();

	constructor(
		private readonly _context: vscode.ExtensionContext
	) { }

    // Implement the necessary methods (open, resolve, etc.)
    openCustomDocument(uri: vscode.Uri, openContext: vscode.CustomDocumentOpenContext, token: vscode.CancellationToken): vscode.CustomDocument {
        // Check if there are existing webviews for the URI
        let existingDocument = this.openDocuments.get(uri.fsPath);
        if (existingDocument) {
            // Dispose the existing document if needed
            return existingDocument;
        }

        // Create a new CustomDocument instance for the URI
        const newDocument = new CustomDocument(uri);
        this.openDocuments.set(uri.fsPath, newDocument);

        return newDocument;
    }

    // Implement the necessary methods (open, resolve, etc.)
    async setWebviewForDocument(document: CustomDocument, webviewPanel: vscode.WebviewPanel): Promise<void> {
        /**
         * This function updates the webview panel with the new document
         * 
         * @param document The document to be displayed in the webview
         * @param webviewPanel The webview panel to be updated
         * 
         * @returns void
         */

        // Send a message to the webview to update the header info
        const documentDataJson = JSON.stringify({
            file: document.file,
            selectedHdu: document.selectedHdu,
            options: document.options,
        });

        // Get the path to the script and style files
        const scriptPath = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'assets/main.js'));
        const stylePath = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'assets/style.css'));
        const webviewUri = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'dist/webview.js'));
        const appIconUri = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'assets/images/appIcon.png'));
        const kipacLogoUri = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'assets/images/KIPAC_stack.png'));
        const codiconsUri = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'node_modules/@vscode/codicons/dist/codicon.css'));
        const codiconsFontUri = webviewPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._context.extensionUri, 'node_modules', '@vscode/codicons', 'dist', 'codicon.ttf'));

        // Generate a nonce to add to the script element
        const randomBytes = crypto.randomBytes(16);
        const nonce = randomBytes.toString('base64');

        // Check if the selected HDU is an image
        const isImage = document.file[`hdu${document.selectedHdu}`].encoded_image;
        const selected = `hdu${document.selectedHdu}`;

        const new_html = `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Security-Policy" content="font-src ${webviewPanel.webview.cspSource} ${codiconsFontUri} vscode-resource: 'unsafe-inline'; style-src ${webviewPanel.webview.cspSource} ${codiconsUri} vscode-resource: 'unsafe-inline'; script-src 'self' 'self' https://*.vscode-cdn.net https://d3js.org https://mpld3.github.io 'unsafe-inline'; img-src 'self' 'self' https://*.vscode-cdn.net data:;">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Your Custom Editor</title>
            <link href="${stylePath}" rel="stylesheet" type="text/css" >
            <link href="${codiconsUri}" rel="stylesheet" type="text/css">
        </head>
        <body>
            <!-- Header Section -->
            <div id="headerContainer">
                <div id="headerContent">
                    <img src="${appIconUri}" alt="App Icon" width="50" height="50">
                    <div>
                        <h2>FitsImageVOY</h2>
                        <p>From DS9 to VOY, dedicated to all Rubin Observatory scientists</p>
                    </div>
                </div>
            </div>
        
            <!-- Main Container -->
            <div id="container">
                <!-- Header Selector and Headers Container -->
                <div id="headerInfoContainer">
                    <div id="optionsSection">
                        <h2>Selectable options</h2>


                        <div id="hduListContainer" class="dropdown-container">
                            <label for="hduSelector">Select HDU:</label>
                            <vscode-dropdown id="hduSelector"></vscode-dropdown>
                            <vscode-data-grid id="basic-grid" aria-label="Basic"></vscode-data-grid>
                        </div>

                        <div id="optionContainer">

                            <vscode-divider role="presentation"></vscode-divider>
                            
                            <div id="selector1"></div>

                            <vscode-divider role="presentation"></vscode-divider>

                            <div id="selector2"></div>

                            <vscode-divider role="presentation"></vscode-divider>

                            <div id="selector3"></div>

                        </div>

                    </div>

                    <div class="section-divider vertical" data-resize="optionsSection|headerInfoSection"></div>
        
                    <!-- Header Info Section -->
                    <div id="headerInfoSection">
                        <div id="headerInfoHeader">
                            <h2>Header Information</h2>
                            <vscode-text-field placeholder="Search" id="input">
                                <span slot="start" appearance="icon" class="codicon codicon-search"></span>
                            </vscode-text-field>
                        </div>
                        <div id="headerInfo"></div>
                    </div>
                </div>
        
                <div class="section-divider horizontal" data-resize="headerInfoContainer|imageSection"></div>
                
                <!-- Image Section -->
                <div id="imageSection">
                    <div id="imageLoadingOverlay" hidden>
                        <span class="codicon codicon-loading codicon-modifier-spin"></span>
                    </div>
                    <div id="imageTableOptionsDiv">
                        <div id="savePngDiv">
                            <vscode-button id="savePngButton" appearance="secondary" onclick="savePng()">Save to PNG</vscode-button>
                            <vscode-text-field type="text" class="saveFields" id="saveFn" value=${path.basename(document.uri.fsPath.replace(/.fit[s]?/gi, ".png"))}>Filename</vscode-text-field>
                            <vscode-text-field type="text" class="saveFields" id="savePath"  value=${path.dirname(document.uri.fsPath)}>Path</vscode-text-field>
                            <vscode-text-field type="text" class="saveFields" id="saveDpi" value="300">DPI</vscode-text-field>
                        </div>
                        <div class="section-divider vertical" data-resize="savePngDiv|imageTableDiv"></div>
                        <div id="imageTableDiv">
                            <div id="imageDiv" class="responsive-plot">
                                ${isImage ? document.file[selected].html_plot : ''}
                            </div>
                        </div>
                    </div>
                    <!-- Add your image or placeholder here -->
                    <!--  <img id="fits-image" alt="FITS Image" src="" onerror="showPlaceholder()"> -->
                    <h2 id="placeholderText" style="display: none;">No image to display.</h2>
                    <h2 id="imageErrorText" style="display: none; color: red;">Error loading image.</h2>
                </div>
            </div>
            <script type="module" src="${webviewUri}"></script>
            <script>
                const vscode = acquireVsCodeApi();

                // Update overlay when image is loading
                function setLoading(isLoading) {
                    const overlay = document.getElementById('imageLoadingOverlay');
                    if (!overlay) return;
                    overlay.hidden = !isLoading;
                }

                // Update the color value in the options
                function updateColor(event) {
                    setLoading(true);
                    // Post a message to VS Code
                    vscode.postMessage({
                        command: 'colormapChanged',
                        newColormap: event,
                    });
                }

                // Update the scale value in the options
                function updateScale(event) {
                    setLoading(true);
                    // Post a message to VS Code
                    vscode.postMessage({
                        command: 'scaleChanged',
                        newScale: event,
                        newClim: { vmin: null, vmax: null } // tells backend to use auto/matplotlib defaults
                    });
                }
                
                // Validate number pattern using regex
                const numberPattern = /^[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?$/;

                function parseValidNumber(text) {
                    if (!numberPattern.test(text)) return null;
                    const n = Number(text);
                    return Number.isFinite(n) ? n : null;
                }

                function validateNumbers() {

                    const vminEl = document.getElementById('vmin');
                    const vmaxEl = document.getElementById('vmax');

                    const vminRaw = vminEl.value.trim();
                    const vmaxRaw = vmaxEl.value.trim();

                    const vmin = vminRaw === '' ? null : parseValidNumber(vminRaw);
                    const vmax = vmaxRaw === '' ? null : parseValidNumber(vmaxRaw);

                    const vminHasValue = vminRaw !== '';
                    const vmaxHasValue = vmaxRaw !== '';

                    let vminInvalid = vminHasValue && vmin === null;
                    let vmaxInvalid = vmaxHasValue && vmax === null;

                    const bothValidNumbers = vmin !== null && vmax !== null;
                    const rangeInvalid = bothValidNumbers && !(vmin < vmax);

                    if (rangeInvalid) {
                        vminInvalid = true;
                        vmaxInvalid = true;
                    }

                    const applyInvalid = (el, isInvalid) => {
                        el.toggleAttribute('invalid', isInvalid);
                        if (isInvalid) {
                            el.setAttribute('aria-invalid', 'true');
                        } else {
                            el.removeAttribute('aria-invalid');
                        }
                    };

                    applyInvalid(vminEl, vminInvalid);
                    applyInvalid(vmaxEl, vmaxInvalid);

                    if (!bothValidNumbers || rangeInvalid) return;

                    return {vmin, vmax};
                }

                // Update the clim value in the options
                function updateClim() {

                    const clim = validateNumbers()

                    if (!clim) return;

                    // If number are valid, update as usual
                    setLoading(true);
                    // Post a message to VS Code
                    vscode.postMessage({
                       command: 'climChanged',
                       newClim: clim
                    });
                }

                function resetClim() {
                    const vminEl = document.getElementById('vmin');
                    const vmaxEl = document.getElementById('vmax');
                    if (vminEl) vminEl.value = '';
                    if (vmaxEl) vmaxEl.value = '';
                    
                    setLoading(true);
                    vscode.postMessage({
                        command: 'climReset',
                        newClim: { vmin: null, vmax: null } // tells backend to use auto/matplotlib defaults
                    });
                }

                function updateCols() {
                    setLoading(true);

                    const xColEl = document.getElementById("xColSelect")
                    const yColEl = document.getElementById("yColSelect")
                    
                    newCols = { x: xColEl.value, y: yColEl.value}
                    vscode.postMessage({
                        command: 'colsChanged',
                        newCols: newCols
                    });
                }
                
                // Update the scale value in the options
                function updateXTableScale(event) {
                    setLoading(true);

                    // Post a message to VS Code
                    vscode.postMessage({
                        command: 'tableXScaleChanged',
                        newXScale: event,
                    });
                }

                function updateYTableScale(event) {
                    setLoading(true);

                    // Post a message to VS Code
                    vscode.postMessage({
                        command: 'tableYScaleChanged',
                        newYScale: event,
                    });
                }
                
                function interactivePointSize(event) {
                    const psv = document.getElementById("pointSizeValue");
                    psv.innerHTML = event;
                }

                function updatePointSize(event) {
                    setLoading(true);
                    vscode.postMessage({
                        command: 'pointSizeChanged',
                        newPointSize: event,
                    });
                }

                function updatePlotType(event) {
                    setLoading(true);
                    vscode.postMessage({
                        command: 'plotTypeChanged',
                        newPlotType: event,
                    });
                }
                
                function savePng() {
                        const saveFn = document.getElementById("saveFn").value;
                        const savePath = document.getElementById("savePath").value;
                        const dpi = document.getElementById("saveDpi").value;

                    vscode.postMessage({
                        command: 'savePng',
                        saveFn: saveFn,
                        savePath: savePath,
                        dpi: dpi
                        })
                }

                // Adjust the scale on page load and whenever the window is resized
                document.addEventListener('DOMContentLoaded',  () => {
                    // Dynamically create a script element for main.js
                    const mainScript = document.createElement('script');
                    mainScript.type = 'module';
                    mainScript.nonce = '${nonce}';
                    mainScript.src = '${scriptPath}';
                    // Append the script element to the document
                    document.body.appendChild(mainScript);

                    // Post a message to VS Code
                    vscode.postMessage({
                        command: 'doneLoading'
                    });
                });
            </script>
        </body>
        <footer>
            <div id="footerContainer">
                <div id="footerText">
                    <p>Designed by <a href="https://github.com/gmegh/" target="_blank">@gmegh</a></p>
                    <p>at KIPAC, Stanford University</p>
                </div>
                
                <div id="kipacLogo">
                    <!-- Replace 'path_to_your_logo.png' with the actual path to your KIPAC logo -->
                    <img src="${kipacLogoUri}" alt="KIPAC Logo" style="max-width: 80px; height: auto;">
                </div>
            </div>
        </footer>
        </html>`;

        webviewPanel.webview.html = new_html;
    }

    async resolveCustomEditor(
        document: CustomDocument,
        webviewPanel: vscode.WebviewPanel,
        _token: vscode.CancellationToken
    ): Promise<void> {
        /**
         * This function resolves the webview panel with the new document
         * 
         * @param document The document to be displayed in the webview
         * @param webviewPanel The webview panel to be updated
         * @param _token The cancellation token
         * 
         * @returns void
         */
        
        // Setup initial content for the webview
        webviewPanel.webview.options = {
            enableScripts: true,
        };
    
        // Call the open method to load the header
        try {
            await document.open();
        } catch (error) {
            vscode.window.showErrorMessage(`FitsImageVOY: failed to open FITS file: ${String(error)}`);
            return;
        }

        // Set the webview for the document
        await this.setWebviewForDocument(document, webviewPanel);

        // Definition for "Dark VSCode Themes"
        const darkKinds = [
            vscode.ColorThemeKind.Dark,
            vscode.ColorThemeKind.HighContrast
        ];

        // Send user theme info to webviewPanel
        function updateTheme() {
          webviewPanel.webview.postMessage({
            command: 'updateTheme',
            data: darkKinds.includes(vscode.window.activeColorTheme.kind)
          });
        }
    
        // update when the user changes theme
        const themeListener = vscode.window.onDidChangeActiveColorTheme(() => updateTheme());
        webviewPanel.onDidDispose(() => themeListener.dispose());

        // Setup for render only last image options
        const COALESCE_MS = 150;
        let coalesceTimer: NodeJS.Timeout | undefined;
        let latestRenderRequestId = 0;

        const buildImagePayload = () =>
            JSON.stringify({
                file: document.file,
                selectedHdu: document.selectedHdu,
                options: document.options
            });

        const scheduleImageRender = (reason: string) => {
            latestRenderRequestId += 1;
            const requestId = latestRenderRequestId;

            if (coalesceTimer) {
                clearTimeout(coalesceTimer);
            }

            coalesceTimer = setTimeout(async () => {
                coalesceTimer = undefined;

                try {
                    await document.open();
                } catch (error) {
                    vscode.window.showErrorMessage(`FitsImageVOY: Encountered rendering error: ${String(error)}`);
                    await webviewPanel.webview.postMessage({
                        command: 'renderError',
                        error: String(error),
                        reason: reason,
                    });
                    return;
                }

                if (requestId !== latestRenderRequestId) {
                    return; // Stale request, drop it
                }

                await webviewPanel.webview.postMessage({
                    command: 'updateImage',
                    data: buildImagePayload(),
                    reason: reason,
                });
            }, COALESCE_MS);
        };

        // Handle messages from the webview
        webviewPanel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    // Handle hdu selected change
                    case 'hduSelectorChanged':
                        document.selectedHdu = message.newSelectedHdu;
                        await this.setWebviewForDocument(document, webviewPanel);
                        break;
                    case 'doneLoading':

                        updateTheme();

                        await webviewPanel.webview.postMessage({
                            command: 'updateHeaderInfo', 
                            data: buildImagePayload() 
                        });

                        await webviewPanel.webview.postMessage({ 
                            command: 'updateImage', 
                            data: buildImagePayload(), 
                            reason: 'initialLoad' 
                        });

                        break;

                    // Handle colormap change
                    case 'colormapChanged':
                        document.options.colormap = message.newColormap;

                        scheduleImageRender(message.command); 

                        break;
                    // Handle scale change
                    case 'scaleChanged':
                        document.options.scale = message.newScale;
                        document.options.clim = message.newClim;

                        scheduleImageRender(message.command);

                        break;

                    // Handle clim change
                    case 'climChanged':
                        document.options.clim = message.newClim;
                        
                        scheduleImageRender(message.command);

                        break;

                    // Handle clim reset

                    case 'climReset':
                        document.options.clim = message.newClim;

                        scheduleImageRender(message.command);

                        break;

                    // Table option changes

                    case 'colsChanged':
                        document.options.cols = message.newCols;

                        scheduleImageRender(message.command);

                        break;

                    case 'tableXScaleChanged':
                        document.options.tableScale.x = message.newXScale;

                        scheduleImageRender(message.command);

                        break;
                    
                    case 'tableYScaleChanged':
                        document.options.tableScale.y = message.newYScale;

                        scheduleImageRender(message.command);

                        break;

                    case 'pointSizeChanged':
                        document.options.pointSize = message.newPointSize;

                        scheduleImageRender(message.command);

                        break;

                    case 'plotTypeChanged':
                        document.options.plotType = message.newPlotType;

                        scheduleImageRender(message.command);

                        break;

                    case 'savePng':
                        document.options.saveMode = true;
                        document.options.saveFn = message.saveFn;
                        document.options.savePath = message.savePath;
                        document.options.dpi = message.dpi;

                        document.generateImageFromFits(document.uri.fsPath, document.options);

                        break;
                }
            },
            undefined,
        );

    }

    saveCustomDocument(document: vscode.CustomDocument, cancellation: vscode.CancellationToken): Thenable<void> {
        // Implement the logic to save the custom document
        return Promise.resolve();
    }
}


class CustomDocument implements vscode.CustomDocument {
    // Implement the necessary methods and properties for your custom document
    private _uri: vscode.Uri;
    private _file: any;  // Replace with the actual type of your header
    private _options: any;
    private _selectedHdu: any;  // Replace with the actual type of your header

    constructor(uri: vscode.Uri) {
        this._uri = uri;

        const configuration = vscode.workspace.getConfiguration('fitsimagevoy');
        const defaultColormap = configuration.get('defaultColormap', 'viridis');
        const defaultScale = configuration.get('defaultScale', 'linear');
        const defaultClim = configuration.get('defaultClim', { vmin: null, vmax: null });
        const defaultCols = configuration.get('defaultCols', { x: null, y: null });
        const defaultTableScale = configuration.get('defaultTableScale', { x: 'linear', y: 'linear'});
        const defaultPointSize = configuration.get('defaultPointSize', 1);
        const defaultPlotType = configuration.get('defaultPlotType', 'line');
        const hduLabels = configuration.get('hduLabels', 'index');
        const defaultSavePath = configuration.get('defaultSavePath', null);
        const defaultDpi = configuration.get('defaultDPI', '300');
        this.options = {
            'colormap': defaultColormap,
            'scale': defaultScale,
            'clim': defaultClim,
            'cols': defaultCols,
            'tableScale': defaultTableScale,
            'pointSize': defaultPointSize,
            'plotType': defaultPlotType,
            'hduLabels': hduLabels,
            'savePath': defaultSavePath,
            'saveFn': null,
            'dpi': defaultDpi
        }; 
    }
    get uri(): vscode.Uri {
        return this._uri;
    }

    get file(): any {
        return this._file;
    }

    set file(value: any) {
        this._file = value;
    }

    get options(): any {
        return this._options;
    }

    set options(value: any) {
        this._options = value;
    }

    get selectedHdu(): any {
        return this._selectedHdu;
    }

    set selectedHdu(value: any) {
        this._selectedHdu = value;
    }

    // Implement the open method to load the header
    async open(): Promise<void> {
        /**
         * This function opens the FITS file and stores the data in the _file variable
         * 
         * @returns void
         */
        
        this._file = await this.generateImageFromFits(this._uri.fsPath, this._options);

        // Initialize a variable to store the selected element
        if (!Number.isInteger(this._selectedHdu) || !this._file[`hdu${this._selectedHdu}`]) {
            this._selectedHdu = 0; // If selectedHdu isn't set or isn't valid, fallback to HDU 0
        

            // Iterate over the elements and find the first one with encoded_image not equal to none
            for (const key in this._file) {
                const element = this._file[key];

                if (element['encoded_image'] !== false) {
                    this._selectedHdu = parseInt(key.replace('hdu', ''), 10);
                    break;
                }
            }
        }
    }

    async generateImageFromFits(fitsFilePath: string, options: RenderOptions): Promise<any> {
        /**
         * This function reads a FITS file and returns the image data, headers and html.
         * It connects with python to run this process.
         * 
         * @param fitsFilePath The path to the FITS file
         * @param colormap The colormap to be used
         * @param scale The scale to be used
         * 
         * @returns A Json string containing the image data, the headers and html.
         */
        return new Promise<any>((resolve, reject) => {

            // Get the Python interpreter path from VSCode settings
            const pythonInterpreterPath: string | undefined = vscode.workspace.getConfiguration('python').get('defaultInterpreterPath');
    
            let command = '';
            if (pythonInterpreterPath === undefined) {
                command = osvar === 'win32' ? 'python' : 'python3';
            } else {
                command = pythonInterpreterPath;
            }

            const pythonScriptPath = path.join(__dirname, 'python', 'generate_image.py');
            const argv = this.buildPythonArgs(pythonScriptPath, fitsFilePath, options)
            .map((arg) => shellQuote(String(arg))).join(' ');
            command = shellQuote(command)

            //const command = osvar === 'win32' ? 'python' : 'python3';
            //const activationCommand = 'source /sdf/group/rubin/sw/w_latest/loadLSST.bash';
            const activationCommand = '';
            const combinedCommand = `
                ${activationCommand} > /dev/null 2>&1 &&
                ${command} ${argv}
            `;

            // Combine activation and Python commands
            const pythonProcess = spawn('bash', ['-c', combinedCommand]);
            
            let result = '';  // Variable to store the base64-encoded image data
            let stderrText = '';
            let settled = false;

            // Prevent accidental duplicate promise resolves/rejects (easier debugging)
            const failOnce = (err: Error) => {
                if (settled) return;
                settled = true;
                reject(err);
            };

            const resolveOnce = (value: any) => {
                if (settled) return;
                settled = true;
                resolve(value);
            };

            if (pythonProcess?.stdout && pythonProcess?.stderr) {

                pythonProcess.stdout.on('data', (data) => {
                    // Handle data received from the Python script
                    result += data.toString();
                });
        
                pythonProcess.stderr.on('data', (data) => {
                    stderrText += data.toString(); // collect, but do not fail yet
                });

                pythonProcess.on('error', (err) => {
                    failOnce(new Error(`Failed to start Python process: ${String(err)}`))
                });
        
                pythonProcess.on('close', (code) => {
                    // Check for positive exit
                    if (code !== 0) {
                        return failOnce(
                            new Error(
                            `Failed to generate image from FITS file (exit ${code}). ${stderrText.trim() || result.trim() || 'No output.'}`
                        ));
                    }

                    try {
                        const output = JSON.parse(result);
                        resolveOnce(output);
                    } catch (error) {
                        failOnce(
                            new Error(
                                `Failed to parse Python output as JSON. ${stderrText.trim() || String(error)}`
                            )
                        )
                    }
                });
            } else {
                failOnce(new Error('Failed to start the Python process.'));
            }
        
        });
    }

    private buildPythonArgs(scriptPath: string, fitsFilePath: string, options: RenderOptions): string[] {

        const args = [scriptPath, fitsFilePath];
        const add = (flag: string, val: unknown) => { if (val !== undefined && val !== null) args.push(flag, String(val)); };
        add("--colormap", options.colormap);
        add("--scale", options.scale);
        add("--vmin", options.clim?.vmin);
        add("--vmax", options.clim?.vmax);
        add("--x_col", options.cols?.x);
        add("--y_col", options.cols?.y);
        add("--table_scale_x", options.tableScale?.x);
        add("--table_scale_y", options.tableScale?.y);
        add("--plot_type", options.plotType);
        add("--point_size", options.pointSize);
        add("--save_path", options.savePath);
        add("--save_fn", options.saveFn);
        add("--dpi", options.dpi);
        if (options.saveMode) {
            args.push("--save_mode");
            args.push("--save_hdu", this._selectedHdu);
            this._options.saveMode = false;
        }

        return args;
    }

	dispose(): void {}
}

// Activate the extension
export function activate(context: vscode.ExtensionContext) {
    const pythonInterpreterPath: string | undefined = vscode.workspace.getConfiguration('python').get('defaultInterpreterPath');
    
    let command = '';
    //const command = osvar === 'win32' ? 'python' : 'python3';
    if (pythonInterpreterPath === undefined) {
        command = osvar === 'win32' ? 'python' : 'python3';
    } else {
        command = pythonInterpreterPath;
    }

    command = shellQuote(command)
    
    //const activationCommand = 'source /sdf/group/rubin/sw/w_latest/loadLSST.bash';
    const activationCommand = '';
    
    const combinedCommand = `
        ${activationCommand} > /dev/null 2>&1 &&
        ${command} -m pip install -r ${requirements}
    `;

    spawn('bash', ['-c', combinedCommand]);

    // Register the custom editor provider
    context.subscriptions.push(
        vscode.window.registerCustomEditorProvider('fitsimagevoy.fits', new CustomEditorProvider(context), 
        {
            webviewOptions: {
				retainContextWhenHidden: true,
			},
        })
    );
}

// Deactivate the extension
export function deactivate() {
    // Clean up resources when the extension is deactivated
}