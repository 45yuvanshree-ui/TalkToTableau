import gradio as gr


APP_TITLE = "TalkToTableau"

APP_DESCRIPTION = """
### TalkToTableau

An Agentic Business Intelligence framework for:

- Natural-language business questions
- Text-to-SQL
- Evidence-based analytics
- Visual analytics
- Data storytelling

The analytical backend will be connected in later steps.
"""


def create_app() -> gr.Blocks:
    """Create the TalkToTableau Gradio application."""

    with gr.Blocks(
        title=APP_TITLE,
    ) as app:
        gr.Markdown(
            f"# {APP_TITLE}"
        )

        gr.Markdown(
            APP_DESCRIPTION
        )

        gr.Markdown(
            "## Ask a Business Question"
        )

        user_question = gr.Textbox(
            label="Business Question",
            placeholder=(
                "Example: What was the total revenue "
                "by region in 2025?"
            ),
            lines=3,
        )

        run_button = gr.Button(
            "Analyze",
            variant="primary",
        )

        status_output = gr.Markdown(
            value=(
                "Enter a business question and click "
                "**Analyze**."
            )
        )

        run_button.click(
            fn=lambda question: (
                "UI shell is working. "
                "Analytical workflow will be connected next."
                if question and question.strip()
                else "Please enter a business question."
            ),
            inputs=user_question,
            outputs=status_output,
        )

    return app


if __name__ == "__main__":
    application = create_app()

    application.launch(
        inbrowser=True,
    )