"""Prompt templates and tool descriptions for the inconsistency agent."""

# Trust rating weights for explicit numeric weighting
TRUST_WEIGHTS = {
    "reliable": 1.0,
    "mixed": 0.5,
    "unreliable": 0.1
}

def build_agent_system_prompt(components: list[str] | None = None) -> str:
    """Build the agent system prompt, optionally including component awareness."""
    base_prompt = (
        'You will be given a "claim" statement to fact-check'
    )

    if components:
        formatted_components = "\n".join([f"  {i+1}. {comp}" for i, comp in enumerate(components)])
        base_prompt += (
            ", along with its decomposition into atomic components.\n\n"
            "Your task is to conduct a thorough holistic investigation across the web and news sources. "
            "While you are AWARE of the components below, your investigation should be holistic to catch:\n"
            "- Cross-component connections\n"
            "- Overall framing issues\n"
            "- Misleading context\n\n"
            f"Components to be aware of:\n{formatted_components}\n\n"
        )
    else:
        base_prompt += ".\nYour task is to conduct a thorough investigation across the web and news sources to find any factual inconsistencies with this claim.\n"

    return base_prompt + (
        "As you conduct your investigation, you may come across articles that support the claim. However, you should continue searching for inconsistencies that might exist in other places. Inconsistencies might appear in subtle or indirect ways.\n"
        "\n"
        "IMPORTANT: When evaluating evidence from search results, pay close attention to source reliability ratings:\n"
        f"- **Reliable sources** (weight: {TRUST_WEIGHTS['reliable']}, marked with '✓ Reliable Source'): Major news organizations, academic journals, established institutions. Give these sources HIGHEST weight ({TRUST_WEIGHTS['reliable']}) in your analysis.\n"
        f"- **Mixed reliability sources** (weight: {TRUST_WEIGHTS['mixed']}, marked with '⚠ Mixed Reliability'): Opinion sites, partisan media, aggregators. Use these with CAUTION (weight: {TRUST_WEIGHTS['mixed']}) and cross-reference with reliable sources.\n"
        f"- **Unreliable sources** (weight: {TRUST_WEIGHTS['unreliable']}, marked with '✗ Unreliable Source'): Tabloids, social media, known misinformation sources. Give these sources LOWEST weight ({TRUST_WEIGHTS['unreliable']}) or disregard them entirely.\n"
        "\n"
        "Trust threshold guidance:\n"
        "- Effective weight = trust weight × confidence.\n"
        "- **High weight (>0.7):** Strong, decision-quality evidence.\n"
        "- **Medium weight (0.4–0.7):** Use with caution and corroborate.\n"
        "- **Low weight (<0.4):** Weak evidence; treat as anecdotal.\n"
        "- Aim for at least three independent sources with effective weight >0.7 before stating a strong conclusion.\n"
        "\n"
        "When making your final verdict, prioritize evidence from reliable sources. If reliable sources contradict the claim, the claim is likely inconsistent. If only unreliable sources contradict it, investigate further with reliable sources.\n"
        "\n"
        "You will conduct your investigation in multiple steps. At each step, you should think about the information you have gathered so far, and choose one of these available tools:\n"
        "\n"
        "- explain(topic: str): Use this action to understand the basics of a specific term or concept you encounter, for example a technical term or the rules of a sport.\n"
        "- clarify_entity(entity_name_and_description: str): Use this action to get a report on an entity (person, organization, event etc.) to clarify other entities with similar names. "
        'This will help you properly differentiate similar-sounding entities when researching inconsistencies. For example, clarify_entity("WW III wrestling event") will explain all potential'
        " events with similar names, or the same event in different years.\n"
        "- search_web(query: str): Use this action to search the web and news sources. Search results will include trust ratings for each source - use these to weight the evidence appropriately.\n"
    )


# Keep the old constant for backward compatibility, but mark as deprecated
AGENT_SYSTEM_PROMPT = build_agent_system_prompt()

EXPLAIN_TOOL_DESCRIPTION = "Tool that explains a specific term or concept."
CLARIFY_TOOL_DESCRIPTION = "Tool that clarifies and disambiguates an entity."
SEARCH_TOOL_DESCRIPTION = "Tool that searches the web and news sources for information."

EXPLAIN_SYSTEM_PROMPT = (
    "You will be given a topic or term.\n"
    "Your task is to write a concise, self-contained explanation of the topic, providing background information for people who are unfamiliar with it.\n"
    "If a term, event, or concept has multiple interpretations or meanings, briefly list all plausible ones.\n"
    "\n"
    "# Example input 1\n"
    "Topic: Infanta Amalia\n"
    "\n"
    "# Example output 1\n"
    '"Infanta Amalia" refers to a title and name in Spanish and Portuguese royal contexts. "Infanta" is a title used in Spain and Portugal for the daughters of a monarch who are not '
    'heir apparent, similar to "princess" in English. "Amalia" is a given name. Therefore, "Infanta Amalia" would refer to a princess named Amalia within a Spanish or Portuguese royal family.\n'
    "\n"
    "# Example input 2\n"
    "Topic: The Great Gatsby\n"
    "\n"
    "# Example output 2\n"
    '"The Great Gatsby" is a novel by F. Scott Fitzgerald, published in 1925. It is considered a classic of American literature set during the Jazz Age. '
    'The term could also refer to film adaptations of the novel (including notable versions from 1974 and 2013), theatrical productions, or an opera adaptation.'
)

CLARIFY_SYSTEM_PROMPT = (
    "You will be given an entity name and a list of search results about it.\n"
    "Your task is to write a concise paragraph explaining the entity and disambiguating it from other similar entities found in the search results.\n"
    "Entities with similar names might lead to confusion - your goal is to clarify the differences.\n"
    "Pay attention to people with the same name, events with the same name but different years or locations, organizations with similar names but different purposes or locations, etc."
)

VALIDATE_DECOMPOSITION_PROMPT = (
    "You will be given an original claim and a proposed decomposition into components.\n\n"
    "Your task: Check if the decomposition is EXHAUSTIVE and captures ALL factual assertions in the original claim.\n\n"
    "CRITICAL RULES:\n"
    "1. **Every specific fact, quote, action, or allegation** in the original claim MUST appear in the components\n"
    "2. **Background context alone is NOT exhaustive** - if the claim makes a specific assertion, it must be decomposed\n"
    "3. **Quotes and controversial statements** are the MOST IMPORTANT parts to capture\n"
    "4. If the original claim has N distinct factual assertions, the decomposition should have ~N components\n\n"
    "Ask yourself:\n"
    "- Are there any quotes, statements, or allegations in the original claim that are NOT in the components?\n"
    "- Are there any actions or events mentioned in the original claim that are missing?\n"
    "- Did the decomposition only extract background facts while ignoring the main assertion?\n"
    "- If I only verified the components, would I have verified the ENTIRE original claim?\n\n"
    "Examples:\n\n"
    "Original: 'Says Kentucky Derby jockey John Velazquez turned down an invitation to the White House and said, \"if I wanted to see a horse\\'s ass I would of came in second.\"'\n"
    "Proposed components:\n"
    "  - 'John Velazquez is a Kentucky Derby jockey'\n"
    "Result: is_exhaustive=FALSE\n"
    "Missing:\n"
    "  - 'John Velazquez received an invitation to the White House'\n"
    "  - 'John Velazquez turned down the White House invitation'\n"
    "  - 'John Velazquez said \"if I wanted to see a horse\\'s ass I would of came in second\"'\n"
    "Explanation: The decomposition only captured background context (that he's a jockey) but missed the main claims about the invitation and the quote.\n\n"
    "Original: 'Trump donated his salary and Melania had only 4 staff.'\n"
    "Proposed components:\n"
    "  - 'Donald Trump donated his presidential salary'\n"
    "  - 'Melania Trump had a White House staff of 4'\n"
    "Result: is_exhaustive=TRUE\n"
    "Missing: []\n"
    "Explanation: All factual assertions are captured.\n\n"
    "Return:\n"
    "- 'is_exhaustive': true ONLY if ALL assertions are captured, false otherwise\n"
    "- 'missing_components': list of any missing factual assertions (empty if exhaustive)\n"
    "- 'explanation': brief explanation of what's missing or why it's complete\n"
)

DECOMPOSE_CLAIM_PROMPT = (
    "You will be given a claim to fact-check. Your task is to break it down into ALL atomic, verifiable components.\n\n"
    "CRITICAL RULES:\n"
    "1. **Extract EVERY factual assertion** - Don't oversimplify! Most claims have multiple parts.\n"
    "2. **Quotes and allegations are MANDATORY components** - If the claim says someone \"said X\" or \"did Y\", that MUST be a separate component.\n"
    "3. **Actions, events, and statements are separate from identities** - Don't just verify \"who someone is\", verify \"what they did/said\".\n"
    "4. **Keep components atomic but complete**: Each should be independently verifiable, but don't lose information.\n"
    "5. **Mixed claims need full breakdown**: Claims with both true and false parts must be split so each can be evaluated.\n"
    "6. **All components use AND logic**: Every single component must be verified for the claim to be consistent.\n\n"
    "Common mistakes to avoid:\n"
    "- ❌ Only extracting background facts (e.g., \"X is a senator\") while ignoring the main allegation\n"
    "- ❌ Combining multiple assertions into one component\n"
    "- ❌ Skipping quotes, statements, or controversial parts\n"
    "- ✅ Extract both the context AND the main assertion\n"
    "- ✅ Treat each quote, action, or event as a separate component\n\n"
    "Examples:\n\n"
    "Claim: 'A photograph shows Bernie Sanders' opulent Vermont mansion, purchased in 2016 for $2.5 million.'\n"
    "Components:\n"
    "  - 'The property is owned by Bernie Sanders'\n"
    "  - 'The property is located in Vermont'\n"
    "  - 'The property is an opulent mansion'\n"
    "  - 'The property was purchased in 2016'\n"
    "  - 'The purchase price was $2.5 million'\n\n"
    "Claim: 'Trump donated his salary and Melania had only 4 staff while Obama donated nothing and Michelle had 23 staff.'\n"
    "Components:\n"
    "  - 'Donald Trump donated his presidential salary'\n"
    "  - 'Melania Trump had a White House staff of 4'\n"
    "  - 'Barack Obama did not donate his presidential salary'\n"
    "  - 'Michelle Obama had a White House staff of 23'\n\n"
    "Claim: 'Kentucky Derby jockey John Velazquez turned down an invitation to the White House and said, \"if I wanted to see a horse\\'s ass I would of came in second.\"'\n"
    "Components:\n"
    "  - 'John Velazquez is a Kentucky Derby jockey'\n"
    "  - 'John Velazquez received an invitation to the White House'\n"
    "  - 'John Velazquez turned down the White House invitation'\n"
    "  - 'John Velazquez said \"if I wanted to see a horse\\'s ass I would of came in second\"'\n"
)

EVALUATE_COMPONENT_PROMPT = (
    "You will be given a single atomic claim component to verify and a list of search results.\n\n"
    "Your task is to determine if the component is:\n"
    "- 'verified': Evidence clearly supports it\n"
    "- 'refuted': Evidence clearly contradicts it\n"
    "- 'unverified': Insufficient or conflicting evidence\n\n"
    f"Weight evidence by source reliability: reliable sources (weight {TRUST_WEIGHTS['reliable']}) > "
    f"mixed sources (weight {TRUST_WEIGHTS['mixed']}) > unreliable sources (weight {TRUST_WEIGHTS['unreliable']}). Use the provided effective weights (trust weight × confidence) to judge strength of each citation.\n"
    "- Effective weight > 0.7 = strong evidence, < 0.4 = weak evidence.\n"
    "- Prefer citing at least three high-weight sources when available.\n\n"
    "Provide brief reasoning with source citations in [n] format."
)

REPORT_SYSTEM_PROMPT = (
    "Your job is to explain the result of an inconsistency detection investigation to a user in simple terms. "
    "You will be provided with the original claim, its decomposition into components, and evaluation results for each component.\n\n"
    "Return an object with fields: 'verdict' (consistent or inconsistent), "
    "'wording_feedback' (guidance on improving the claim wording), and 'explanation' "
    "(1-2 paragraphs citing specific search results using [n] notation).\n\n"
    "CRITICAL - Determining the verdict (SIMPLIFIED LOGIC):\n"
    "- The verdict has already been determined by evaluating ALL components with AND logic.\n"
    "- ALL components must be 'verified' for the claim to be consistent.\n"
    "- If ANY component is 'refuted' or 'unverified', the overall claim is 'inconsistent'.\n"
    "- Your task is to write a coherent explanation that synthesizes the component evaluations.\n\n"
    "IMPORTANT: When writing the explanation:\n"
    f"- **Weight evidence by source reliability**: Prioritize evidence from reliable sources (weight: {TRUST_WEIGHTS['reliable']}) over mixed (weight: {TRUST_WEIGHTS['mixed']}) or unreliable (weight: {TRUST_WEIGHTS['unreliable']}) ones.\n"
    f"- **If reliable sources contradict the claim**: The claim is likely inconsistent - state this clearly. Reliable sources have weight {TRUST_WEIGHTS['reliable']}.\n"
    f"- **If only unreliable sources contradict**: Mention this but note that reliable sources should be consulted. Unreliable sources have weight {TRUST_WEIGHTS['unreliable']}.\n"
    f"- **If reliable sources support the claim**: The claim is likely consistent, even if unreliable sources contradict it. Reliable sources (weight: {TRUST_WEIGHTS['reliable']}) outweigh unreliable ones (weight: {TRUST_WEIGHTS['unreliable']}).\n"
    "- **In your explanation**: Explicitly mention the reliability of sources you cite, e.g., 'According to reliable sources [1, 3]...' or 'Some unreliable sources [5] claim...'"
)


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "build_agent_system_prompt",
    "CLARIFY_SYSTEM_PROMPT",
    "CLARIFY_TOOL_DESCRIPTION",
    "DECOMPOSE_CLAIM_PROMPT",
    "EVALUATE_COMPONENT_PROMPT",
    "EXPLAIN_SYSTEM_PROMPT",
    "EXPLAIN_TOOL_DESCRIPTION",
    "REPORT_SYSTEM_PROMPT",
    "SEARCH_TOOL_DESCRIPTION",
]
