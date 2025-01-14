const { createServer } = require('http');
const { parse } = require('url');
const path = require('path');

// Get paths
const staticDir = path.join(__dirname, 'static');
// Add node_modules to Node's module search path
process.env.NODE_PATH = path.join(staticDir, 'node_modules');
require('module').Module._initPaths();

const next = require(path.join(staticDir, 'node_modules/next'));

const dev = process.env.NODE_ENV !== 'production';
// Get host and port from environment variables set by __main__.py
const hostname = process.env.HOST || 'localhost';
const port = parseInt(process.env.PORT, 10) || 3000;

// Configure Next.js
const app = next({
    dev,
    dir: staticDir,
    hostname,
    port
});

const handle = app.getRequestHandler();

app.prepare().then(() => {
    createServer(async (req, res) => {
        try {
            const parsedUrl = parse(req.url, true);
            await handle(req, res, parsedUrl);
        } catch (err) {
            console.error('Error occurred handling', req.url, err);
            res.statusCode = 500;
            res.end('internal server error');
        }
    })
    .once('error', (err) => {
        console.error(err);
        process.exit(1);
    })
    .listen(port, hostname, () => {
        console.log(`> Ready on http://${hostname}:${port}`);
    });
});
