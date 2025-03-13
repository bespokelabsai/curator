"""Generate a large dataset of mathematical concepts through a sequence of LLM calls.

This example demonstrates how to generate a large dataset of mathematical concepts through a sequence of LLM calls.
It shows how to define a custom curator.LLM class for each generation step, and how to use the curator.LLM.batch method to generate the dataset.
"""

# ruff: noqa

import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("concept_generation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Define models for each step in the generation pipeline
class ConceptCategory(BaseModel):
    """Category and subcategory information for a mathematical concept"""

    main_category: str = Field(description="Main mathematical category (e.g., Number Theory, Calculus)")
    subcategory: str = Field(description="Specific subcategory within the main category")
    difficulty: str = Field(description="Difficulty level: Basic, Intermediate, or Advanced")
    visual_potential: int = Field(description="Rating from 1-10 on how well this concept can be visualized")
    audience: str = Field(description="Target audience for this concept (e.g., High School, Undergraduate)")
    related_fields: List[str] = Field(description="Other fields of mathematics or science this connects to")


class ConceptIdea(BaseModel):
    """Core idea for a mathematical concept animation"""

    title: str = Field(description="Clear, specific title for the concept")
    core_question: str = Field(description="Intriguing question that captures the essence of the concept")
    key_insight: str = Field(description="The main mathematical insight or 'aha' moment")
    visualization_approach: str = Field(description="Brief description of how this could be visually presented")
    tags: List[str] = Field(description="5-8 relevant tags for categorizing this concept")


class ConceptOutline(BaseModel):
    """Detailed outline for a mathematical concept"""

    script_outline: List[str] = Field(description="Bullet-point outline of the script structure (5-8 points)")
    script_excerpt: str = Field(description="A brief excerpt from the script introducing the concept")
    visual_elements: List[Dict] = Field(description="List of key visual elements to include in the animation")
    equations: List[str] = Field(description="Important equations that would be featured in the animation")
    key_insights: List[str] = Field(description="Key takeaways or insights from exploring this concept")
    prerequisites: List[str] = Field(description="Concepts the viewer should understand beforehand")


# Define the curator.LLM classes for each generation step
class CategoryGenerator(curator.LLM):
    """Generates diverse mathematical categories and subcategories"""

    response_format = ConceptCategory

    def prompt(self, input: Dict) -> str:
        return f"""
        Generate a specific, interesting mathematical category and subcategory combination that would be suitable for a mathematical animation.
        
        Your task is to go beyond common, generic categories and identify specific, focused areas that offer rich visual and conceptual potential.
        
        CONSTRAINTS:
        - The category combination should be SPECIFIC (not just "Calculus" but something like "Applications of Integration in Physics")
        - It should have strong VISUAL POTENTIAL (can be illustrated well through animation)
        - It should offer opportunities for mathematical INSIGHT (not just computation)
        - The combination should be at the {input['target_difficulty']} difficulty level
        - Consider the intended audience: {input.get('audience', 'High school students')}
        - IMPORTANT: ONLY include topics that are taught in HIGH SCHOOL mathematics (Algebra, Geometry, Trigonometry, Pre-Calculus, Basic Calculus)
        
        AVOID:
        - Overly common or generic categories
        - Categories with limited visual representation potential
        - College or graduate-level topics
        - Abstract or theoretical concepts beyond high school level
        
        EXAMPLES OF GOOD SPECIFIC CATEGORIES:
        - "Geometric Interpretation of the Quadratic Formula"
        - "Visualizing Trigonometric Identities"
        - "Applications of Similar Triangles"
        - "Understanding Exponential Growth Visually"
        - "The Geometry Behind the Pythagorean Theorem"
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "main_category": "The broad field (e.g., Algebra, Geometry, Trigonometry)",
            "subcategory": "The specific topic within that field",
            "difficulty": "{input['target_difficulty']}",
            "visual_potential": 8,  // A number from 1-10
            "audience": "High school students",
            "related_fields": ["Field1", "Field2", "Field3"]  // 2-4 related fields
        }}
        """

    def parse(self, input: Dict, response: ConceptCategory) -> Dict:
        return {
            "main_category": response.main_category,
            "subcategory": response.subcategory,
            "difficulty": response.difficulty,
            "visual_potential": response.visual_potential,
            "audience": response.audience,
            "related_fields": response.related_fields,
            "target_difficulty": input["target_difficulty"],  # Pass through the input difficulty
        }


class ConceptIdeaGenerator(curator.LLM):
    """Generates core concept ideas based on mathematical categories"""

    response_format = ConceptIdea

    def prompt(self, input: Dict) -> str:
        return f"""
        Create an interesting, specific concept for a mathematical animation in the style of 3Blue1Brown.
        
        CATEGORY: {input['main_category']}
        SUBCATEGORY: {input['subcategory']}
        DIFFICULTY: {input['difficulty']}
        AUDIENCE: {input['audience']}
        
        Your concept should:
        1. Focus on building INTUITION rather than formal proofs
        2. Have strong VISUAL potential (rated {input['visual_potential']}/10 for visualization)
        3. Contain an "aha!" moment or mathematical insight
        4. Be specific enough for a 5-10 minute video
        5. Connect to related fields: {', '.join(input['related_fields'])}
        
        The best 3Blue1Brown videos reveal unexpected connections, build intuition through visualization, or offer a new perspective on familiar concepts.
        
        EXAMPLES OF GOOD TITLES:
        - "The Hidden Patterns in Prime Spirals"
        - "Why π^π^π^π is an Integer (and why that's surprising)"
        - "The Geometric Meaning of the Determinant"
        
        Make your concept interesting and specific, not generic.
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "title": "A specific, engaging title for the animation",
            "core_question": "The intriguing question that the animation explores",
            "key_insight": "The main mathematical insight or 'aha' moment",
            "visualization_approach": "How this concept can be visualized effectively",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]  // 5-8 relevant tags
        }}
        """

    def parse(self, input: Dict, response: ConceptIdea) -> Dict:
        return {
            "main_category": input["main_category"],
            "subcategory": input["subcategory"],
            "difficulty": input["difficulty"],
            "audience": input["audience"],
            "related_fields": input["related_fields"],
            "title": response.title,
            "core_question": response.core_question,
            "key_insight": response.key_insight,
            "visualization_approach": response.visualization_approach,
            "tags": response.tags,
        }


class ConceptDetailGenerator(curator.LLM):
    """Generates detailed outlines for mathematical concepts"""

    response_format = ConceptOutline

    def prompt(self, input: Dict) -> str:
        return f"""
        Create a simplified script outline for a short (30-second) mathematical animation.
        
        CONCEPT:
        Title: {input['title']}
        Core Question: {input['core_question']}
        Key Insight: {input['key_insight']}
        
        CATEGORY: {input['main_category']} / {input['subcategory']}
        DIFFICULTY: {input['difficulty']}
        AUDIENCE: {input['audience']}
        
        Develop a SIMPLE script outline and visual elements that:
        1. Focus on ONE key idea or insight
        2. Can be explained in 30 seconds
        3. Use only basic visual elements (text, simple shapes, arrows, 1-2 equations)
        4. Avoid complex animations or 3D visualizations
        
        Your outline should be BRIEF (3-4 points maximum) and focus on the most essential aspect of the concept.
        The visual elements should be SIMPLE and easy to implement in manim.
        
        Focus on the absolute core of the concept - what's the ONE thing you want viewers to understand?
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "script_outline": [
                "Point 1: Brief introduction (5 seconds)",
                "Point 2: Show the key concept (15 seconds)",
                "Point 3: Reveal the insight (10 seconds)"
            ],
            "script_excerpt": "A brief 2-3 sentence introduction to the concept...",
            "visual_elements": [
                {{
                    "name": "Simple Element 1",
                    "description": "A basic visual element (text, shape, or equation)"
                }},
                {{
                    "name": "Simple Element 2",
                    "description": "Another basic visual element"
                }}
            ],
            "equations": [
                "Simple equation 1"
            ],
            "key_insights": [
                "The single most important insight about this concept"
            ],
            "prerequisites": [
                "Basic prerequisite 1"
            ]
        }}
        
        Ensure your response includes ALL of these fields with the exact field names as shown above.
        The field names must be lowercase with underscores as shown.
        KEEP IT SIMPLE - this is for a 30-second animation!
        """

    def parse(self, input: Dict, response: ConceptOutline) -> Dict:
        """Parse the response from the LLM.

        Args:
            input: The input dictionary
            response: The response from the LLM

        Returns:
            The parsed response
        """
        full_concept = {
            **input,
            "script_outline": response.script_outline,
            "script_excerpt": response.script_excerpt,
            "visual_elements": response.visual_elements,
            "equations": response.equations,
            "key_insights": response.key_insights,
            "prerequisites": response.prerequisites,
        }
        return full_concept


def generate_math_concepts(num_concepts=10000, output_dir="math_concepts_dataset"):
    """Generate a large dataset of mathematical concepts through a sequence of LLM calls.

    Args:
        num_concepts: Number of concepts to generate
        output_dir: Directory to save the dataset

    Returns:
        Path to the generated dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting generation of {num_concepts} mathematical concepts")

    # Initialize generators
    category_generator = CategoryGenerator(model_name="gpt-4o-mini")
    idea_generator = ConceptIdeaGenerator(model_name="gpt-4o")
    detail_generator = ConceptDetailGenerator(model_name="gpt-4o")

    # Define difficulty distribution - focus heavily on basic concepts for simpler animations
    difficulty_distribution = {
        "Basic": 0.7,  # 70% basic concepts
        "Intermediate": 0.3,  # 30% intermediate concepts
        "Advanced": 0.0,  # 0% advanced concepts - removed entirely
    }
    # Step 1: Create inputs for category generation
    logger.info("Preparing inputs for category generation")

    category_inputs = []
    for _ in range(num_concepts):
        # Assign difficulty based on distribution
        difficulties = list(difficulty_distribution.keys())
        weights = list(difficulty_distribution.values())
        target_difficulty = random.choices(difficulties, weights=weights, k=1)[0]

        # Set audience to high school only
        audience = "High School"

        category_inputs.append({"target_difficulty": target_difficulty, "audience": audience})

    # Create Hugging Face dataset for categories
    category_input_dataset = Dataset.from_list(category_inputs)

    # Step 2: Generate categories all at once
    logger.info("Generating mathematical categories")
    categories_dataset = category_generator(category_input_dataset)

    # Step 3: Generate concept ideas all at once
    logger.info("Generating concept ideas based on categories")
    concept_ideas_dataset = idea_generator(categories_dataset)

    # Step 4: Generate detailed concepts all at once
    logger.info("Generating detailed concept outlines")
    detailed_concepts_dataset = detail_generator(concept_ideas_dataset)

    # Step 5: Save the final dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = os.path.join(output_dir, f"math_concepts_dataset_{timestamp}")
    detailed_concepts_dataset.save_to_disk(dataset_path)

    # Also save as JSON for easier inspection
    concepts_list = [dict(concept) for concept in detailed_concepts_dataset]
    json_path = os.path.join(output_dir, f"math_concepts_{timestamp}.json")

    with open(json_path, "w") as f:
        json.dump(concepts_list, f, indent=2)

    logger.info(f"Generated {len(concepts_list)} mathematical concepts")
    logger.info(f"Dataset saved to {dataset_path}")
    logger.info(f"JSON saved to {json_path}")

    # Return the path to the dataset
    return dataset_path


def analyze_dataset(dataset_path, output_dir="."):
    """Analyze the diversity of the generated dataset.

    Args:
        dataset_path: Path to the dataset
        output_dir: Directory to save the analysis

    Returns:
        Dictionary with analysis results
    """
    # Load dataset
    dataset = Dataset.load_from_disk(dataset_path)

    # Analyze category distribution
    categories = {}
    for item in dataset:
        main_category = item["main_category"]
        if main_category in categories:
            categories[main_category] += 1
        else:
            categories[main_category] = 1

    # Analyze difficulty distribution
    difficulties = {}
    for item in dataset:
        difficulty = item["difficulty"]
        if difficulty in difficulties:
            difficulties[difficulty] += 1
        else:
            difficulties[difficulty] = 1

    # Analyze audience distribution
    audiences = {}
    for item in dataset:
        audience = item["audience"]
        if audience in audiences:
            audiences[audience] += 1
        else:
            audiences[audience] = 1

    # Count unique subcategories
    unique_subcategories = set()
    for item in dataset:
        unique_subcategories.add(item["subcategory"])

    # Analyze tag distribution
    tag_counts = {}
    for item in dataset:
        for tag in item["tags"]:
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

    # Sort results
    categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])

    # Compile analysis
    analysis = {
        "total_concepts": len(dataset),
        "unique_main_categories": len(categories),
        "unique_subcategories": len(unique_subcategories),
        "unique_tags": len(tag_counts),
        "category_distribution": categories,
        "difficulty_distribution": difficulties,
        "audience_distribution": audiences,
        "top_tags": top_tags,
    }

    # Save analysis
    analysis_file = os.path.join(output_dir, "dataset_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Dataset analysis saved to {analysis_file}")

    return analysis


if __name__ == "__main__":
    # For small test run, use a small number of concepts
    test_mode = True
    num_concepts = 100 if test_mode else 10000

    # Generate the dataset
    dataset_path = generate_math_concepts(num_concepts=num_concepts, output_dir="math_concepts_10k")

    # Analyze the dataset
    analysis = analyze_dataset(dataset_path, output_dir="math_concepts_10k")

    print("\nDataset generation complete!")
    print(f"Total concepts: {analysis['total_concepts']}")
    print(f"Unique main categories: {analysis['unique_main_categories']}")
    print(f"Unique subcategories: {analysis['unique_subcategories']}")

    top_categories = list(analysis["category_distribution"].items())[:5]
    print("\nTop 5 categories:")
    for category, count in top_categories:
        print(f"- {category}: {count} concepts ({count/analysis['total_concepts']*100:.1f}%)")
