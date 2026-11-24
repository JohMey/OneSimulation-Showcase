#!/usr/bin/env node
// OneSimulation MCP starter.
//
// OneSimulation's MCP server ships inside the OneSimulation runtime, which is
// installed on the user's machine. This starter contains none of it. When the
// runtime is installed it starts that server and passes everything through.
// When it is not, it still answers: it searches the public example library
// and explains how to get and connect the runtime.
'use strict';

const { existsSync, readFileSync } = require('node:fs');
const { spawn } = require('node:child_process');
const { join } = require('node:path');
const readline = require('node:readline');

const config = JSON.parse(readFileSync(join(__dirname, 'config.json'), 'utf8'));
// A client that leaves an optional setting empty may pass the placeholder itself.
const configured = (process.env.ONESIMULATION_MCP || '').trim();
const runtime = configured.startsWith('${') ? '' : configured;

if (runtime && existsSync(runtime)) {
  // Pass-through: the installed server owns stdin and stdout from here on.
  const child = spawn(runtime, [], { stdio: 'inherit', env: process.env });
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
  child.on('error', (error) => { process.stderr.write(`OneSimulation MCP server could not be started: ${error.message}\n`); process.exit(1); });
  child.on('exit', (code, signal) => process.exit(signal ? 1 : code ?? 0));
} else {
  if (runtime) process.stderr.write(`ONESIMULATION_MCP points to a file that does not exist: ${runtime}\n`);
  serve();
}

function setupText() {
  return [
    `${config.name} is not connected yet. It is a Python library for physics simulation with a local MCP server; this starter only launches it.`,
    '',
    '1. Get the runtime:',
    `   - Plans and trial: ${config.urls.pricing}`,
    `   - Free Education plan for students and teachers: ${config.urls.contact}`,
    `2. Install and activate it: ${config.urls.quickstart}`,
    '   Install the wheel with the "agent" extra. OneSimulation is not on PyPI; never install a package of that name from a package index.',
    '3. Point this starter to the installed server and to a workspace folder:',
    '   ONESIMULATION_MCP=/absolute/path/to/.venv-onesimulation/bin/onesimulation-mcp',
    '   ONESIMULATION_WORKSPACE=/absolute/path/to/a/folder/for/your/projects',
    '   In a client that installed this bundle, set both in the bundle settings and restart the client.',
    '',
    `Meanwhile, search_examples finds worked examples with reference checks in the public library: ${config.urls.examples}`,
  ].join('\n');
}

let library;
async function examples() {
  if (!library) {
    const response = await fetch(config.urls.llms, { signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(`the example library answered HTTP ${response.status}`);
    const text = await response.text();
    library = [];
    let subject = '';
    for (const line of text.split('\n')) {
      const heading = /^### (.+)$/.exec(line);
      if (heading) { subject = heading[1]; continue; }
      const entry = /^- \[(.+?)\]\((https?:\/\/[^)\s]+)\): (.+)$/.exec(line);
      if (entry && subject) library.push({ subject, title: entry[1], url: entry[2], summary: entry[3] });
    }
  }
  return library;
}

async function searchExamples(query, limit) {
  const words = String(query).toLowerCase().split(/[^a-z0-9]+/).filter((word) => word.length > 1);
  if (!words.length) return 'Give one or more words that describe the problem, for example "heat sink forced convection".';
  const scored = (await examples()).map((item) => {
    const title = item.title.toLowerCase();
    const rest = `${item.subject} ${item.summary}`.toLowerCase();
    let score = 0;
    for (const word of words) score += (title.includes(word) ? 3 : 0) + (rest.includes(word) ? 1 : 0);
    return { item, score };
  }).filter((entry) => entry.score > 0).sort((a, b) => b.score - a.score).slice(0, limit);
  if (!scored.length) return `No example matches "${query}". Browse the library: ${config.urls.examples}`;
  return scored.map(({ item }) => `${item.title} (${item.subject})\n${item.summary}\n${item.url}`).join('\n\n')
    + '\n\nEach page has the model, the reference check and the notebook download. Running an example needs the OneSimulation runtime; call setup_onesimulation for the steps.';
}

const TOOLS = [
  {
    name: 'search_examples',
    description: `Search the public ${config.name} example library (${config.examples} worked physics simulations with reference checks) for examples that match a problem description. Returns titles, summaries and links.`,
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Words that describe the physical problem, e.g. "thermal stress turbine blade".' },
        limit: { type: 'integer', minimum: 1, maximum: 20, default: 5, description: 'Number of examples to return.' },
      },
      required: ['query'],
    },
  },
  {
    name: 'setup_onesimulation',
    description: `Explain how to get the ${config.name} runtime and connect its MCP server, which builds, runs and reads simulations on this machine.`,
    inputSchema: { type: 'object', properties: {} },
  },
];

function serve() {
  const send = (message) => process.stdout.write(JSON.stringify(message) + '\n');
  const reply = (id, result) => send({ jsonrpc: '2.0', id, result });
  const fail = (id, code, message) => send({ jsonrpc: '2.0', id, error: { code, message } });
  const text = (value, isError = false) => ({ content: [{ type: 'text', text: value }], isError });

  readline.createInterface({ input: process.stdin }).on('line', async (line) => {
    if (!line.trim()) return;
    let message;
    try { message = JSON.parse(line); } catch { return fail(null, -32700, 'Parse error'); }
    const { id, method, params } = message;
    if (id === undefined || id === null) return; // notifications need no answer
    try {
      if (method === 'initialize') {
        return reply(id, {
          protocolVersion: params?.protocolVersion || '2025-06-18',
          capabilities: { tools: {} },
          serverInfo: { name: 'onesimulation-starter', title: `${config.name} starter`, version: config.version },
          instructions: `The ${config.name} runtime is not connected. Use search_examples to find worked examples and setup_onesimulation for the setup steps.`,
        });
      }
      if (method === 'ping') return reply(id, {});
      if (method === 'tools/list') return reply(id, { tools: TOOLS });
      if (method === 'tools/call') {
        const args = params?.arguments || {};
        if (params?.name === 'setup_onesimulation') return reply(id, text(setupText()));
        if (params?.name === 'search_examples') {
          const limit = Number.isInteger(args.limit) ? Math.min(20, Math.max(1, args.limit)) : 5;
          try { return reply(id, text(await searchExamples(args.query ?? '', limit))); }
          catch (error) { return reply(id, text(`The example library could not be read: ${error.message}. Browse it at ${config.urls.examples}`, true)); }
        }
        return fail(id, -32602, `Unknown tool: ${params?.name}`);
      }
      return fail(id, -32601, `Method not found: ${method}`);
    } catch (error) {
      return fail(id, -32603, String(error?.message || error));
    }
  });
}
