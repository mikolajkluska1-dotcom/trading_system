import streamlit as st

def render_hud():
    st.components.v1.html(
        """
        <div id="hud-bar" style="
            display:flex;
            align-items:center;
            gap:18px;
            padding:8px 16px;
            margin-bottom:16px;
            border:1px solid var(--accent,#1f2937);
            border-radius:10px;
            background:rgba(255,255,255,0.03);
            font-size:12px;
            color:var(--text,#d1d5db);
        ">
            <span>CONNECTING HUD…</span>
        </div>

        <script>
        const hud = document.getElementById("hud-bar");
        const ws = new WebSocket("ws://127.0.0.1:8001/ws/hud");


        function color(v){
            if(v >= 80) return "#7f1d1d";
            if(v >= 60) return "#92400e";
            return "var(--text,#d1d5db)";
        }

        ws.onmessage = (e) => {
            const d = JSON.parse(e.data);
            const isOps = ["ADMIN","ROOT"].includes(d.role);

            let html = "";

            if(isOps){
                html += `<span>NODE <b>${d.node}</b></span>`;
                html += `<span>USER <b>${d.user}</b></span>`;
                html += `<span style="color:${color(d.cpu)}">CPU <b>${d.cpu.toFixed(1)}%</b></span>`;
                html += `<span style="color:${color(d.mem)}">MEM <b>${d.mem.toFixed(1)}%</b></span>`;
            }

            html += `<span>FUNDS <b>$${Number(d.funds).toLocaleString()}</b></span>`;
            html += `<span>TIME <b>${d.time}</b></span>`;

            hud.innerHTML = html;
        };

        ws.onerror = () => {
            hud.innerHTML = "<span style='color:#92400e'>HUD ERROR</span>";
        };

        ws.onclose = () => {
            hud.innerHTML = "<span style='color:#92400e'>HUD DISCONNECTED</span>";
        };
        </script>
        """,
        height=56,
    )
