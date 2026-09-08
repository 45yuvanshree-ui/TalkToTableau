from pydantic import BaseModel, Field


class StorySection(BaseModel):
    """A section of the generated business story."""

    heading: str
    content: str


class BusinessStory(BaseModel):
    """Executive-friendly business narrative based on verified evidence."""

    summary: str = Field(
        description=(
            "A concise summary of the most important verified finding."
        )
    )

    sections: list[StorySection] = Field(
        default_factory=list,
        description="Structured sections explaining the analysis."
    )

    visualization_explanation: str = Field(
        description=(
            "A simple explanation of what the visualization shows, "
            "when a visualization is available."
        )
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence-based recommendations. Leave empty when "
            "the available evidence does not justify recommendations."
        )
    )