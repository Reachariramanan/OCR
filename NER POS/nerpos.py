import gradio as gr
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from flair.data import Sentence
from flair.models import SequenceTagger

# --- 1) BERT NER setup (∼108M params) ---
tokenizer_ner = AutoTokenizer.from_pretrained("mdroth/bert-finetuned-ner-accelerate")
model_ner     = AutoModelForTokenClassification.from_pretrained("mdroth/bert-finetuned-ner-accelerate")
ner_pipe      = pipeline(
    "ner",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple"
)

def recognize_entities(text: str):
    ents = ner_pipe(text)
    # Prepare rows for the DataFrame
    rows = [[e["entity_group"], e["word"], e["start"], e["end"]] for e in ents]
    if not rows:
        rows = [["—", "No entities found", "—", "—"]]
    # Prepare pretty JSON
    json_out = json.dumps(
        [
            {"entity": e["entity_group"], "text": e["word"], "start": e["start"], "end": e["end"]}
            for e in ents
        ] or [],
        indent=2
    )
    return rows, json_out

# --- 2) Flair POS setup (∼4.48M params) ---
pos_tagger = SequenceTagger.load("flair/pos-english")

def pos_tag_with_error(text: str):
    sentence = Sentence(text)
    try:
        pos_tagger.predict(sentence)
        tokens = [
            {"token": token.text, "pos": token.get_label("pos").value}
            for token in sentence
        ]
        pretty = json.dumps(tokens, indent=2)
        return pretty, gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return (
            "[]",
            gr.update(visible=True),
            gr.update(visible=True, value=str(e))
        )

# Example sentence to load into both inputs
EXAMPLE = (
    "In a landmark hearing, the Supreme Court of India examined a challenge "
    "brought by the Bharatya Janata Party against the new electoral funding "
    "rules, citing possible constitutional violations."
)

# --- 3) Build the Gradio app with JSON outputs for both NER & POS ---
with gr.Blocks() as demo:
    gr.Markdown("""
    ## ⚙️ Pravartak Foundation's NLP Playground
    """)

    # Single button to populate example text
    example_btn = gr.Button("Use Example Sentence")

    with gr.Row():
        # --- Left column: NER ---
        with gr.Column():
            gr.Markdown("### Named Entity Recognition")
            gr.Markdown("**Model:** `bert-finetuned-ner-108M` (~108M params)")
            ner_input = gr.Textbox(lines=5, placeholder="Enter text for NER…")
            ner_out_df   = gr.Dataframe(
                headers=["Entity", "Text", "Start", "End"],
                label="Detected Entities (Table)"
            )
            ner_out_json = gr.Textbox(
                label="Detected Entities (JSON)",
                interactive=False,
                show_copy_button=True
            )
            ner_input.submit(
                fn=recognize_entities,
                inputs=[ner_input],
                outputs=[ner_out_df, ner_out_json]
            )
            gr.Button("Run NER").click(
                fn=recognize_entities,
                inputs=[ner_input],
                outputs=[ner_out_df, ner_out_json]
            )

        # --- Right column: POS ---
        with gr.Column():
            gr.Markdown("### Part‑of‑Speech Tagging")
            gr.Markdown("**Base Model:** `flair/pos-english` (~4.48M params)")
            gr.Markdown(
            "**Description:** The Flair POS English model uses stacked Flair embeddings "
            "(2048 dimensions each, combined to 4096 input size), with a Bi‑LSTM of one layer "
            "and 256 hidden units, totaling approximately 4.48 million parameters."
            )
            pos_input    = gr.Textbox(lines=2, placeholder="Enter text for POS tagging…")
            pos_out      = gr.Textbox(
                label="Token → POS (JSON)",
                interactive=False,
                show_copy_button=True
            )
            show_err_btn = gr.Button("Show Error", visible=False)
            err_box      = gr.Textbox(
                lines=4,
                label="Error Traceback",
                visible=False,
                interactive=False
            )
            run_pos_btn = gr.Button("Run POS Tagger")
            run_pos_btn.click(
                fn=pos_tag_with_error,
                inputs=[pos_input],
                outputs=[pos_out, show_err_btn, err_box]
            )
            show_err_btn.click(
                fn=lambda err: gr.update(visible=True),
                inputs=[err_box],
                outputs=[err_box]
            )

    # Wire up “Use Example” button after inputs are defined
    example_btn.click(
        fn=lambda: (EXAMPLE, EXAMPLE),
        inputs=[],
        outputs=[ner_input, pos_input]
    )

if __name__ == "__main__":
    demo.launch()
