// FailSight agent for Node / Express: records time, route, method, status and latency of every request and posts them in batches to <monitorUrl>/ingest/api.
// No dependencies (needs Node 18+ for the global fetch). Usage:
//   const failsight = require('./failsight-express');
//   app.use(failsight({ monitorUrl: 'http://127.0.0.1:8000', appId: 'my-app', apiKey: process.env.FAILSIGHT_KEY }));
function failsight({ monitorUrl, appId, apiKey, flushMs = 1000, maxBatch = 500, fetchImpl = globalThis.fetch }) {
  let buffer = [];
  async function flush() {
    if (!buffer.length) return;
    const events = buffer.splice(0, maxBatch);
    try {
      await fetchImpl(`${monitorUrl.replace(/\/$/, '')}/ingest/api`, {
        method: 'POST',
        headers: Object.assign({ 'Content-Type': 'application/json' }, apiKey ? { 'X-API-Key': apiKey } : {}),
        body: JSON.stringify({ system: appId, events }),
      });
    } catch (e) { /* the monitor being down must never break the app: drop the batch */ }
  }
  const timer = setInterval(flush, flushMs);
  if (timer.unref) timer.unref();
  const middleware = function (req, res, next) {
    const start = process.hrtime.bigint();
    const wall = Date.now() / 1000;
    res.on('finish', () => {
      buffer.push({
        timestamp: wall,
        method: req.method,
        route: (req.route && req.route.path) || (req.baseUrl || '') + (req.path || req.url || ''),
        status_code: res.statusCode,
        latency_ms: Number(process.hrtime.bigint() - start) / 1e6,
        content_length: Number((res.get && res.get('content-length')) || 0),
      });
      if (buffer.length >= maxBatch) flush();
    });
    next();
  };
  middleware.flush = flush;
  return middleware;
}
module.exports = failsight;
