"""HTML templates for the web interface."""

HTML_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <title>Unlock</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      width: 100%;
      background-color: #000;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      width: 100%;
    }
    .status {
      text-align: center;
      margin-bottom: 30px;
      min-height: 88px;
    }
    #typed {
      display: inline-block;
      letter-spacing: 10px;
      font-size: 36px;
      font-weight: 400;
      min-height: 44px;
      line-height: 44px;
      transition: opacity 0.2s ease-in-out;
    }
    #msg {
      font-size: 16px;
      margin-top: 10px;
      color: #bbb;
      min-height: 20px;
    }
    .keypad {
      display: grid;
      grid-template-columns: repeat(3, 100px);
      grid-gap: 20px;
      justify-content: center;
      align-content: center;
    }
    button.key {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      border: none;
      font-size: 32px;
      font-weight: 500;
      color: #fff;
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(10px);
      cursor: pointer;
      transition: background 0.15s, transform 0.1s;
    }
    button.key:active {
      transform: scale(0.94);
      background: rgba(255, 255, 255, 0.25);
    }
    button.action {
      font-size: 18px;
      background: transparent;
      color: #bbb;
      text-transform: uppercase;
      letter-spacing: 1px;
      border: none;
      margin-top: 20px;
      cursor: pointer;
    }
    .mode {
      position: absolute;
      top: 15px;
      left: 15px;
      font-size: 14px;
      color: #bbb;
    }
    .mode label {
      margin-right: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="mode">
      Mode:
      <label><input type="radio" name="mode" value="train" checked> train</label>
      <label><input type="radio" name="mode" value="test"> test</label>
    </div>
    <div class="status">
      <div id="typed"></div>
      <div id="msg"></div>
    </div>
    <div class="keypad">
      <button class="key">1</button>
      <button class="key">2</button>
      <button class="key">3</button>
      <button class="key">4</button>
      <button class="key">5</button>
      <button class="key">6</button>
      <button class="key">7</button>
      <button class="key">8</button>
      <button class="key">9</button>
      <div></div>
      <button class="key">0</button>
      <div></div>
    </div>
    <button id="undo" class="action">Undo</button>
    <button id="abort" class="action">Abort</button>
  </div>

  <script>
    const typed = document.getElementById('typed');
    const msg = document.getElementById('msg');

    function currentMode(){
      const el = document.querySelector('input[name="mode"]:checked');
      return el ? el.value : 'train';
    }

    function setMsg(t){ msg.textContent = t; }

    async function sendKey(d){
      const res = await fetch('/api/key', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({digit: String(d), mode: currentMode()})
      });
      const j = await res.json();
      typed.textContent = j.typed || '';
      setMsg(j.message || '');
      if ((j.count || 0) >= 4) {
        typed.textContent = '';
      }
    }

    async function undo(){
      const res = await fetch('/api/undo', {method:'POST'});
      const j = await res.json();
      typed.textContent = j.typed || '';
      setMsg(j.message || '');
    }

    async function abort(){
      const res = await fetch('/api/abort', {method:'POST'});
      const j = await res.json();
      typed.textContent = '';
      setMsg(j.message || '');
    }

    document.querySelectorAll('button.key').forEach(b => {
      b.addEventListener('click', () => sendKey(b.textContent.trim()));
    });
    document.getElementById('undo').addEventListener('click', undo);
    document.getElementById('abort').addEventListener('click', abort);
  </script>
</body>
</html>
"""