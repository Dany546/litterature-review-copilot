// extension.ts - Main VS Code Extension Entry Point
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Comment Provider for Sidebar
class CommentProvider implements vscode.TreeDataProvider<CommentItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<CommentItem | undefined | null | void> = new vscode.EventEmitter<CommentItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<CommentItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor(private workspaceRoot: string) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: CommentItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: CommentItem): Promise<CommentItem[]> {
        if (!this.workspaceRoot) {
            return [];
        }

        if (!element) {
            // Root level - show documents
            const documentsPath = path.join(this.workspaceRoot, 'documents');
            if (!fs.existsSync(documentsPath)) {
                return [];
            }

            const files = fs.readdirSync(documentsPath)
                .filter(file => file.endsWith('.md'));

            return files.map(file => {
                const filePath = path.join(documentsPath, file);
                const comments = this.extractComments(filePath);
                const label = `${file} (${comments.human.length + comments.ai.length})`;
                
                return new CommentItem(
                    label,
                    vscode.TreeItemCollapsibleState.Collapsed,
                    'document',
                    filePath,
                    undefined
                );
            });
        } else if (element.type === 'document') {
            // Show comment categories
            const comments = this.extractComments(element.filePath!);
            
            return [
                new CommentItem(
                    `🙋 Human Comments (${comments.human.length})`,
                    vscode.TreeItemCollapsibleState.Collapsed,
                    'category',
                    element.filePath,
                    'human'
                ),
                new CommentItem(
                    `🤖 AI Comments (${comments.ai.length})`,
                    vscode.TreeItemCollapsibleState.Collapsed,
                    'category',
                    element.filePath,
                    'ai'
                )
            ];
        } else if (element.type === 'category') {
            // Show individual comments
            const comments = this.extractComments(element.filePath!);
            const categoryComments = element.category === 'human' ? comments.human : comments.ai;
            
            return categoryComments.map((comment, index) => {
                const preview = this.getCommentPreview(comment);
                const label = `${index + 1}. ${preview}`;
                
                return new CommentItem(
                    label,
                    vscode.TreeItemCollapsibleState.None,
                    'comment',
                    element.filePath,
                    element.category,
                    comment
                );
            });
        }

        return [];
    }

    private extractComments(filePath: string): { human: any[], ai: any[] } {
        const content = fs.readFileSync(filePath, 'utf-8');
        const humanPattern = /<!--\s*HUMAN_COMMENT:\s*(.+?)\s*-->/gs;
        const aiPattern = /<!--\s*AI_COMMENT:\s*(.+?)\s*-->/gs;

        const human: any[] = [];
        const ai: any[] = [];

        let match;
        while ((match = humanPattern.exec(content)) !== null) {
            try {
                human.push(JSON.parse(match[1]));
            } catch {
                human.push({ text: match[1] });
            }
        }

        while ((match = aiPattern.exec(content)) !== null) {
            try {
                ai.push(JSON.parse(match[1]));
            } catch {
                ai.push({ text: match[1] });
            }
        }

        return { human, ai };
    }

    private getCommentPreview(comment: any): string {
        if (comment.target_section) {
            return `${comment.target_section}: ${(comment.explanation || comment.text || '').substring(0, 50)}...`;
        }
        return (comment.explanation || comment.text || 'No content').substring(0, 60) + '...';
    }
}

class CommentItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly type: 'document' | 'category' | 'comment',
        public readonly filePath?: string,
        public readonly category?: 'human' | 'ai',
        public readonly comment?: any
    ) {
        super(label, collapsibleState);

        if (type === 'comment') {
            this.tooltip = this.createTooltip();
            this.contextValue = category === 'ai' ? 'aiComment' : 'humanComment';
            this.command = {
                command: 'literatureReview.showCommentDetail',
                title: 'Show Comment',
                arguments: [this]
            };
        } else if (type === 'document') {
            this.contextValue = 'document';
        }
    }

    private createTooltip(): string {
        if (!this.comment) return '';
        
        let tooltip = '';
        if (this.comment.target_section) {
            tooltip += `Section: ${this.comment.target_section}\n\n`;
        }
        if (this.comment.explanation) {
            tooltip += `Explanation: ${this.comment.explanation}\n\n`;
        }
        if (this.comment.related_concepts) {
            tooltip += `Related: ${this.comment.related_concepts.join(', ')}\n\n`;
        }
        if (this.comment.timestamp) {
            tooltip += `Time: ${new Date(this.comment.timestamp).toLocaleString()}`;
        }
        return tooltip || this.comment.text || '';
    }
}

// Python Backend Interface
class RAGBackend {
    private pythonPath: string;
    private scriptPath: string;

    constructor(workspaceRoot: string) {
        this.pythonPath = 'python'; // Or configure from settings
        this.scriptPath = path.join(workspaceRoot, 'copilot.py');
    }

    async ingestDocument(filePath: string): Promise<void> {
        const command = `${this.pythonPath} "${this.scriptPath}" ingest "${filePath}"`;
        try {
            const { stdout, stderr } = await execAsync(command);
            console.log(stdout);
            if (stderr) console.error(stderr);
        } catch (error) {
            throw new Error(`Failed to ingest document: ${error}`);
        }
    }

    async generateComments(mdPath: string): Promise<void> {
        const command = `${this.pythonPath} "${this.scriptPath}" process "${mdPath}"`;
        try {
            const { stdout, stderr } = await execAsync(command);
            console.log(stdout);
            if (stderr) console.error(stderr);
        } catch (error) {
            throw new Error(`Failed to generate comments: ${error}`);
        }
    }

    async search(query: string, mode: string = 'hybrid'): Promise<string> {
        const command = `${this.pythonPath} "${this.scriptPath}" search "${query}" ${mode}`;
        try {
            const { stdout } = await execAsync(command);
            return stdout;
        } catch (error) {
            throw new Error(`Search failed: ${error}`);
        }
    }
}

// Main Extension Activation
export function activate(context: vscode.ExtensionContext) {
    console.log('Literature Review Copilot is now active!');

    const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('Please open a workspace folder');
        return;
    }

    const backend = new RAGBackend(workspaceRoot);
    const commentProvider = new CommentProvider(workspaceRoot);

    // Register Tree View
    const treeView = vscode.window.createTreeView('literatureReviewComments', {
        treeDataProvider: commentProvider
    });

    // Watch for file changes
    const watcher = vscode.workspace.createFileSystemWatcher('**/documents/*.md');
    watcher.onDidChange(() => commentProvider.refresh());
    watcher.onDidCreate(() => commentProvider.refresh());
    watcher.onDidDelete(() => commentProvider.refresh());

    // Command: Ingest Document
    const ingestCommand = vscode.commands.registerCommand('literatureReview.ingestDocument', async () => {
        const files = await vscode.window.showOpenDialog({
            canSelectFiles: true,
            canSelectMany: false,
            filters: {
                'Documents': ['pdf', 'docx', 'pptx']
            }
        });

        if (files && files[0]) {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Ingesting document...',
                cancellable: false
            }, async (progress) => {
                try {
                    await backend.ingestDocument(files[0].fsPath);
                    vscode.window.showInformationMessage('Document ingested successfully!');
                    commentProvider.refresh();
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to ingest: ${error}`);
                }
            });
        }
    });

    // Command: Generate AI Comments
    const generateCommentsCommand = vscode.commands.registerCommand('literatureReview.generateComments', async (item?: CommentItem) => {
        let mdPath: string | undefined;

        if (item && item.filePath) {
            mdPath = item.filePath;
        } else {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'markdown') {
                mdPath = editor.document.uri.fsPath;
            }
        }

        if (!mdPath) {
            vscode.window.showErrorMessage('Please select a markdown document');
            return;
        }

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Generating AI comments...',
            cancellable: false
        }, async (progress) => {
            try {
                await backend.generateComments(mdPath!);
                vscode.window.showInformationMessage('AI comments generated!');
                commentProvider.refresh();
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to generate comments: ${error}`);
            }
        });
    });

    // Command: Search Documents
    const searchCommand = vscode.commands.registerCommand('literatureReview.search', async () => {
        const query = await vscode.window.showInputBox({
            prompt: 'Enter your search query',
            placeHolder: 'e.g., transformer architecture'
        });

        if (!query) return;

        const mode = await vscode.window.showQuickPick(
            ['hybrid', 'local', 'global', 'naive'],
            { placeHolder: 'Select search mode' }
        );

        if (!mode) return;

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Searching...',
            cancellable: false
        }, async (progress) => {
            try {
                const result = await backend.search(query, mode);
                
                // Show results in new document
                const doc = await vscode.workspace.openTextDocument({
                    content: `# Search Results\n\nQuery: "${query}"\nMode: ${mode}\n\n${result}`,
                    language: 'markdown'
                });
                vscode.window.showTextDocument(doc);
            } catch (error) {
                vscode.window.showErrorMessage(`Search failed: ${error}`);
            }
        });
    });

    // Command: Show Comment Detail
    const showCommentCommand = vscode.commands.registerCommand('literatureReview.showCommentDetail', async (item: CommentItem) => {
        if (!item.comment) return;

        const panel = vscode.window.createWebviewPanel(
            'commentDetail',
            'Comment Detail',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = getCommentDetailHTML(item.comment, item.category!);
    });

    // Command: Accept AI Comment
    const acceptCommentCommand = vscode.commands.registerCommand('literatureReview.acceptComment', async (item: CommentItem) => {
        if (item.category !== 'ai' || !item.filePath || !item.comment) return;

        try {
            const content = fs.readFileSync(item.filePath, 'utf-8');
            
            // Convert AI_COMMENT to HUMAN_COMMENT
            const aiCommentStr = `<!-- AI_COMMENT: ${JSON.stringify(item.comment, null, 2)} -->`;
            const humanComment = { ...item.comment, accepted: true, accepted_at: new Date().toISOString() };
            const humanCommentStr = `<!-- HUMAN_COMMENT: ${JSON.stringify(humanComment, null, 2)} -->`;
            
            const newContent = content.replace(aiCommentStr, humanCommentStr);
            fs.writeFileSync(item.filePath, newContent, 'utf-8');
            
            vscode.window.showInformationMessage('Comment accepted!');
            commentProvider.refresh();
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to accept comment: ${error}`);
        }
    });

    // Command: Reject AI Comment
    const rejectCommentCommand = vscode.commands.registerCommand('literatureReview.rejectComment', async (item: CommentItem) => {
        if (item.category !== 'ai' || !item.filePath || !item.comment) return;

        try {
            const content = fs.readFileSync(item.filePath, 'utf-8');
            const aiCommentStr = `<!-- AI_COMMENT: ${JSON.stringify(item.comment, null, 2)} -->`;
            const newContent = content.replace(aiCommentStr, '');
            
            fs.writeFileSync(item.filePath, newContent, 'utf-8');
            
            vscode.window.showInformationMessage('Comment rejected!');
            commentProvider.refresh();
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to reject comment: ${error}`);
        }
    });

    // Command: Refresh Comments
    const refreshCommand = vscode.commands.registerCommand('literatureReview.refresh', () => {
        commentProvider.refresh();
    });

    // Register all commands
    context.subscriptions.push(
        ingestCommand,
        generateCommentsCommand,
        searchCommand,
        showCommentCommand,
        acceptCommentCommand,
        rejectCommentCommand,
        refreshCommand,
        treeView,
        watcher
    );

    // Status Bar Item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(book) Lit Review";
    statusBarItem.command = 'literatureReview.search';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
}

function getCommentDetailHTML(comment: any, category: string): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comment Detail</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
        }
        h2 {
            color: ${category === 'ai' ? '#4CAF50' : '#2196F3'};
        }
        .section {
            margin: 20px 0;
            padding: 15px;
            background-color: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 5px;
        }
        .label {
            font-weight: bold;
            color: var(--vscode-textLink-foreground);
        }
        .content {
            margin-top: 5px;
            white-space: pre-wrap;
        }
        .tag {
            display: inline-block;
            padding: 3px 8px;
            margin: 2px;
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            border-radius: 3px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h2>${category === 'ai' ? '🤖 AI Comment' : '🙋 Human Comment'}</h2>
    
    ${comment.target_section ? `
    <div class="section">
        <div class="label">Target Section:</div>
        <div class="content">${comment.target_section}</div>
    </div>
    ` : ''}
    
    ${comment.explanation ? `
    <div class="section">
        <div class="label">Explanation:</div>
        <div class="content">${comment.explanation}</div>
    </div>
    ` : ''}
    
    ${comment.related_concepts && comment.related_concepts.length > 0 ? `
    <div class="section">
        <div class="label">Related Concepts:</div>
        <div class="content">
            ${comment.related_concepts.map((c: string) => `<span class="tag">${c}</span>`).join('')}
        </div>
    </div>
    ` : ''}
    
    ${comment.questions && comment.questions.length > 0 ? `
    <div class="section">
        <div class="label">Questions:</div>
        <div class="content">
            ${Array.isArray(comment.questions) 
                ? comment.questions.map((q: string) => `• ${q}`).join('<br>') 
                : comment.questions}
        </div>
    </div>
    ` : ''}
    
    ${comment.text ? `
    <div class="section">
        <div class="label">Content:</div>
        <div class="content">${comment.text}</div>
    </div>
    ` : ''}
    
    ${comment.timestamp ? `
    <div class="section">
        <div class="label">Timestamp:</div>
        <div class="content">${new Date(comment.timestamp).toLocaleString()}</div>
    </div>
    ` : ''}
</body>
</html>`;
}

export function deactivate() {}
