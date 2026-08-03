"""Build-time student-trait (persona) generation.

Moved from the runtime (src/tutorsim/student.py) when frozen personas were
embedded in the moments release (schema v2, 2026-07-03): released moments
carry the exact persona under student.trait, the runtime only consumes it,
and new moments sets get their personas here at build time.

Public API:
    generate_trait(transcript_prefix, mode, *, model_client, ...) -> str
    generate_traits_for_moments(moments, *, model_client, model_name) -> int
"""

import datetime
import logging
from typing import Optional

from tutorsim_build.resources import resource_text

logger = logging.getLogger(__name__)

_TRAIT_GEN_DIR = "prompts/trait_generator"
_DIMENSIONS_DIR = f"{_TRAIT_GEN_DIR}/dimensions"

# The production trait mode (the paper's `trait`/`oracle` alias).
DEFAULT_TRAIT_MODE = "joined-3"

# ---------------------------------------------------------------------------
# Dimension registry
# ---------------------------------------------------------------------------

ENGAGEMENT_STATIC_DIMENSIONS: dict = {
    "distractedness": None,
    "active_vs_passive_learning": None,
    "affect": None,
}

ENGAGEMENT_CHANGE_DIMENSIONS: dict = {}

COGNITIVE_STATIC_DIMENSIONS: dict = {
    "misconceptions": None,
}

COGNITIVE_CHANGE_DIMENSIONS: dict = {
    "learning_efficiency": None,
}

ALL_ENGAGEMENT_DIMENSIONS: dict = {
    **ENGAGEMENT_STATIC_DIMENSIONS,
    **ENGAGEMENT_CHANGE_DIMENSIONS,
}

ALL_COGNITIVE_DIMENSIONS: dict = {
    **COGNITIVE_STATIC_DIMENSIONS,
    **COGNITIVE_CHANGE_DIMENSIONS,
}

ALL_DIMENSIONS: dict = {**ALL_ENGAGEMENT_DIMENSIONS, **ALL_COGNITIVE_DIMENSIONS}

ALL_DIMENSION_NAMES: list = list(ALL_DIMENSIONS.keys()) + [
    "all",
    "all_engagement",
    "all_cognitive",
    "joined",
    "joined_misconceptions_distractedness",
    "joined_misconceptions_affect",
]


def _load_dimension_text(name: str) -> str:
    """Load and return the dimension description text from package resources."""
    return resource_text(f"{_DIMENSIONS_DIR}/{name}.txt").rstrip("\n")


def get_dimension_description(persona_dimension: str) -> str:
    """Return the text description for a single dimension or aggregate group.

    Aggregate names ("all", "all_engagement", "all_cognitive") compose
    numbered descriptions. "joined" forms are handled in TraitGenerator,
    not here.
    """
    if persona_dimension not in ALL_DIMENSION_NAMES:
        raise ValueError(
            f"Invalid persona dimension: {persona_dimension}. "
            f"Must be one of: {ALL_DIMENSION_NAMES}"
        )

    if persona_dimension in [
        "joined",
        "joined_misconceptions_distractedness",
        "joined_misconceptions_affect",
    ]:
        raise ValueError(
            f"'{persona_dimension}' should not call get_dimension_description()"
        )

    if persona_dimension == "all":
        texts = sorted(_load_dimension_text(n) for n in ALL_DIMENSIONS.keys())
        return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    if persona_dimension == "all_engagement":
        texts = [
            get_dimension_description(n)
            for n in sorted(ALL_ENGAGEMENT_DIMENSIONS.keys())
        ]
        return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    if persona_dimension == "all_cognitive":
        texts = [
            get_dimension_description(n)
            for n in sorted(ALL_COGNITIVE_DIMENSIONS.keys())
        ]
        return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    return _load_dimension_text(persona_dimension)


# ---------------------------------------------------------------------------
# Adapter (wraps tutorsim.client.ModelClient)
# ---------------------------------------------------------------------------


class _ModelWrapperAdapter:
    """Thin shim around tutorsim ModelClient mimicking TraitGenerator's model interface.

    TraitGenerator calls self.model.call(non_system_messages, system_prompt, **kw) -> str.
    This adapter translates that into client.generate(user_text, ..., cacheable_prefix=...).
    """

    def __init__(self, client) -> None:
        self._client = client

    def call(
        self,
        non_system_messages: list,
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **_kwargs,
    ) -> str:
        """Send a single user-message + system-prompt to the wrapped ModelClient.

        system_prompt forwarded as cacheable_prefix (prompt cache friendly).
        """
        user_text = "\n\n".join(
            m.get("content", "")
            for m in (non_system_messages or [])
            if m.get("role") in (None, "user")
        )
        resp = self._client.generate(
            user_text,
            json_mode=False,
            max_tokens=max_tokens or 1024,
            cacheable_prefix=system_prompt or None,
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# TraitGenerator (live path only)
# ---------------------------------------------------------------------------


class TraitGenerator:
    """Generate student traits from example conversations.

    Live generation path only: generate_trait() for single + joined modes.
    """

    def __init__(self, model: _ModelWrapperAdapter) -> None:
        self.model = model

    def get_trait_system_prompt(
        self, trait_type: str, num_sentences: Optional[int] = None
    ) -> str:
        """Build the system prompt for a single dimension.

        Loads prompts/trait_generator/system.txt and substitutes
        {dimension_description} and {sentence_count_suffix}.
        """
        dimension_description = get_dimension_description(trait_type)

        template = resource_text(f"{_TRAIT_GEN_DIR}/system.txt").rstrip("\n")

        if num_sentences is None:
            suffix = ""
        elif num_sentences > 1:
            suffix = f" The description should be {num_sentences} sentences long."
        else:
            suffix = " The description should be 1 sentence long."

        return template.replace(
            "{dimension_description}", dimension_description
        ).replace("{sentence_count_suffix}", suffix)

    def _get_user_prompt(
        self, conversation_text: str, num_sentences: Optional[int] = None
    ) -> str:
        """Build the user prompt for trait generation.

        Loads prompts/trait_generator/user.txt and substitutes
        {conversation_text} and {num_sentences}.
        """
        template = resource_text(f"{_TRAIT_GEN_DIR}/user.txt").rstrip("\n")
        return template.replace("{conversation_text}", conversation_text).replace(
            "{num_sentences}", str(num_sentences)
        )

    def parse_trait_output(self, raw_output: str) -> tuple:
        """Parse raw model output to extract description after 'DESCRIPTION:' marker.

        Returns (description, raw_output).
        """
        if "DESCRIPTION:" in raw_output:
            parts = raw_output.split("DESCRIPTION:", 1)
            description = parts[1].strip()
        else:
            description = raw_output.strip()
        return description, raw_output

    def parse_trait_type(self, trait_type: str) -> tuple:
        """Parse trait mode to (base_trait_type, num_sentences)."""
        if "-" not in trait_type:
            base_trait_type = trait_type
            num_sentences = None
        else:
            base_trait_type, num_sentences_str = trait_type.split("-", 1)
            num_sentences = int(num_sentences_str)

        if base_trait_type not in ALL_DIMENSION_NAMES:
            raise ValueError(
                f"Invalid base trait type: {base_trait_type}. "
                f"Must start with one of: {ALL_DIMENSION_NAMES}"
            )

        if num_sentences is not None and num_sentences <= 0:
            raise ValueError(
                f"Invalid number of sentences: {num_sentences}. Must be greater than 0."
            )

        return base_trait_type, num_sentences

    def _generate_joined_trait(
        self,
        conversation_text: str,
        num_sentences: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        individual_dimensions: Optional[list] = None,
    ) -> str:
        """Generate one trait per dimension, then sort + join."""
        if individual_dimensions is None:
            individual_dimensions = list(ALL_DIMENSIONS.keys())

        dimension_traits: dict = {}
        for dimension in individual_dimensions:
            dimension_mode = dimension
            if num_sentences is not None:
                dimension_mode = f"{dimension}-{num_sentences}"

            trait = self.generate_trait(
                conversation_text=conversation_text,
                trait_mode=dimension_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            dimension_traits[dimension] = trait

        return "\n\n".join(sorted(dimension_traits.values()))

    def generate_trait(
        self,
        conversation_text: str,
        trait_mode: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a student trait description from a conversation."""
        trait_type, num_sentences = self.parse_trait_type(trait_mode)

        if trait_type not in ALL_DIMENSION_NAMES:
            raise ValueError(
                f"Invalid trait mode: {trait_mode}. "
                f"Must start with one of: {ALL_DIMENSION_NAMES}"
            )

        # Joined modes: generate per-dimension, then consolidate.
        if trait_type in [
            "joined",
            "joined_misconceptions_distractedness",
            "joined_misconceptions_affect",
        ]:
            if trait_type == "joined_misconceptions_distractedness":
                individual_dimensions = ["misconceptions", "distractedness"]
            elif trait_type == "joined_misconceptions_affect":
                individual_dimensions = ["misconceptions", "affect"]
            else:  # "joined"
                individual_dimensions = list(ALL_DIMENSIONS.keys())

            return self._generate_joined_trait(
                conversation_text=conversation_text,
                num_sentences=num_sentences,
                temperature=temperature,
                max_tokens=max_tokens,
                individual_dimensions=individual_dimensions,
            )

        # Single-dimension path.
        system_prompt = self.get_trait_system_prompt(trait_type, num_sentences)
        user_prompt = self._get_user_prompt(conversation_text, num_sentences)
        non_system_messages = [{"role": "user", "content": user_prompt}]

        trait_response = self.model.call(
            non_system_messages=non_system_messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        description, _raw = self.parse_trait_output(trait_response)
        return description


# ---------------------------------------------------------------------------
# Module-level wrappers
# ---------------------------------------------------------------------------


def generate_trait(
    transcript_prefix: str,
    mode: str = DEFAULT_TRAIT_MODE,
    *,
    model_client,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a student trait description from a transcript prefix."""
    adapter = _ModelWrapperAdapter(model_client)
    generator = TraitGenerator(adapter)
    return generator.generate_trait(
        conversation_text=transcript_prefix,
        trait_mode=mode,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _format_transcript_prefix(context: list) -> str:
    """Format moment context turns into 'Turn N. ROLE: text' lines.

    Must match the runtime's conversation prefix format exactly — the paper's
    personas were generated from this same rendering of the pre-cut turns.
    """
    return "\n".join(
        f"Turn {t['turn_number']}. {t['role'].upper()}: {t['text']}" for t in context
    )


def generate_traits_for_moments(
    moments: list,
    *,
    model_client,
    model_name: str,
    trait_mode: str = DEFAULT_TRAIT_MODE,
) -> int:
    """Attach a student.trait to every moment that lacks one.

    Personas are generated from the PRE-cut context only (never the reference
    continuation), matching the paper's trait provenance. Moments that already
    carry a trait — e.g. rebuilt from a reference run — are left untouched.

    Returns the number of traits generated.
    """
    generated = 0
    for moment in moments:
        if (moment.student.get("trait") or {}).get("persona"):
            continue
        prefix = _format_transcript_prefix(moment.context)
        persona = generate_trait(prefix, trait_mode, model_client=model_client)
        moment.student["trait"] = {
            "persona": persona.strip(),
            "trait_mode": trait_mode,
            "generator_model": model_name,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }
        generated += 1
        logger.info("trait generated for %s (%d chars)", moment.id, len(persona))
    return generated
