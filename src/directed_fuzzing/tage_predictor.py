from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from .bp import BP
#from ..bp.predictors import ConditionalTAGE
from bp.predictors import ConditionalTAGE


import atexit

@dataclass
class TAGEEvent:
    eid: int
    address: int
    hit_table_id: int
    prediction: bool
    confidence: int
    taken: bool

current_iid = -1
logs_dict: Dict[int, List[TAGEEvent]] = {}
atexit.register(lambda: plot_interactive(logs_dict))

class TageBP(BP):
    def __init__(self, num_state_bits, init_state_val, num_base_entries):
        self.tage = ConditionalTAGE()

        global logs_dict
        logs_dict = {}
        self.counter = 0

    def predict(self, address: int, update_state: bool = False) -> bool:
        return bool(self.tage.predict(address, None)[0])

    def update(self, address: int, taken: bool) -> None:
        prediction, provider_index, confidence, altpred, altpred_index, altpred_confidence = self.tage.predict(address, None)
        self.tage.update(address, None, None, int(taken),
                         prediction, provider_index, confidence, altpred, altpred_index, altpred_confidence)
        self.tage.update_phr(address, 0) # TODO: Need To take into account the real target address of the branch!
        if current_iid not in logs_dict:
            logs_dict[current_iid] = []
        logs_dict[current_iid].append(TAGEEvent(self.counter, address, provider_index, bool(prediction), confidence, taken))
        self.counter += 1

    def snapshot(self) -> Tuple[int, ...]:
        return ()
        raise NotImplemented()

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_tage_logs(events):
    """
    events: list[TAGEEvent] – all from ONE input
    """

    # ---- group events by PC ----
    by_pc = defaultdict(list)
    for e in events:
        by_pc[e.address].append(e)

    fig, ax = plt.subplots(figsize=(14, 6))

    # ---- draw per-PC lines ----
    for pc, evs in by_pc.items():
        evs.sort(key=lambda e: e.eid)

        xs = [e.eid for e in evs]
        ys = [e.confidence for e in evs]

        ax.plot(
            xs,
            ys,
            color="black",
            linewidth=0.5,
            alpha=0.5,
            zorder=1
        )

    # ---- draw points ----
    for e in events:
        correct = (e.prediction == e.taken)
        color = "green" if correct else "red"

        ax.scatter(
            e.eid,
            e.confidence,
            color=color,
            s=80,
            edgecolors="black",
            zorder=2
        )

        # ---- table id inside point ----
        ax.text(
            e.eid,
            e.confidence,
            str(e.hit_table_id),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=3,
            fontweight="bold"
        )

    # ---- axes & labels ----
    ax.set_xlabel("Event ID (test progression)")
    ax.set_ylabel("Confidence (Saturating Counter)")
    ax.set_title("TAGE Confidence Evolution (Per-Address)")

    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig("tage_plot.jpg", dpi=300, bbox_inches="tight")





import plotly.graph_objects as go
from collections import defaultdict

def plot_interactive(events_by_input):
    def max_table_per_input(events_by_input):
        return {
                input_id: max(e.hit_table_id for e in events)
                for input_id, events in events_by_input.items()
                }
    fig = go.Figure()

    for input_id, events in events_by_input.items():
        by_pc = defaultdict(list)

        for e in events:
            by_pc[e.address].append(e)

        for pc, evs in by_pc.items():
            evs.sort(key=lambda e: e.eid)

            xs = [e.eid for e in evs]
            ys = [e.confidence for e in evs]

            colors = [
                "green" if e.prediction == e.taken else "red"
                for e in evs
            ]

            hover = [
                f"""
                Input: {input_id}<br>
                PC: 0x{e.address:x}<br>
                Event: {e.eid}<br>
                Confidence: {e.confidence}<br>
                Table: {e.hit_table_id}<br>
                Predicted: {e.prediction}<br>
                Taken: {e.taken}
                """
                for e in evs
            ]

            max_table = max_table_per_input(events_by_input)[input_id]
            legend_name = f"Input {input_id} [Tmax={max_table}]"


            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers+text",
                    text=[str(e.hit_table_id) for e in evs],
                    textposition="middle center",

                    marker=dict(
                        color=colors,
                        size=10,
                        line=dict(color="black", width=1),
                    ),
                    line=dict(color="black", width=0.5),

                    hovertext=hover,
                    hoverinfo="text",

                    name=legend_name,
                    legendgroup=f"input_{input_id}",
                    visible="legendonly",
                    showlegend=(pc == list(by_pc.keys())[0]),
                )
            )

    fig.update_layout(
        title="BPU Confidence Evolution",
        xaxis_title="Event ID",
        yaxis_title="Confidence",
        legend_title="Inputs",
        hovermode="closest",
        yaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=1
            )
    )

    fig.show()
    fig.write_html("bpu_interactive.html")



