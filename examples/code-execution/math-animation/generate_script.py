import json
import logging
import os
from typing import Dict, List

from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

from bespokelabs import curator

# ruff: noqa
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("math_content_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Define models for hierarchical generation
class Subject(BaseModel):
    """Main mathematical subject area"""

    index: int = Field(description="Index of the mathematical subject")
    name: str = Field(description="Name of the mathematical subject")
    description: str = Field(description="Brief description of this mathematical subject")


class Subjects(BaseModel):
    """List of mathematical subjects"""

    subjects: List[Subject] = Field(description="List of mathematical subjects")


class Topic(BaseModel):
    """Topic within a mathematical subject"""

    index: int = Field(description="Index of the mathematical topic")
    name: str = Field(description="Name of the mathematical topic")
    description: str = Field(description="Brief description of what this topic covers")
    subject: str = Field(description="The parent subject this topic belongs to")


class Topics(BaseModel):
    """List of mathematical topics"""

    topics: List[Topic] = Field(description="List of mathematical topics")


class Question(BaseModel):
    """Interesting mathematical question about a topic"""

    index: int = Field(description="Index of the mathematical question")
    question: str = Field(description="The core mathematical question")
    brief_insight: str = Field(description="Brief insight into the answer or approach")
    visualization_potential: str = Field(description="How this could be visually represented")
    subject: str = Field(description="The parent subject")
    topic: str = Field(description="The specific topic this question addresses")


class Questions(BaseModel):
    """List of mathematical questions"""

    questions: List[Question] = Field(description="List of mathematical questions")


class Script(BaseModel):
    """Full animation script for a mathematical question"""

    title: str = Field(description="Title for the animation")
    narration: str = Field(description="Complete narration script for the animation")
    visual_elements: List[Dict] = Field(description="Visual elements to include in the animation")
    equations: List[str] = Field(description="Mathematical equations to display")
    key_timestamps: Dict[str, str] = Field(description="Key moments in the animation with timestamps")
    visual_style: str = Field(description="Description of the visual style for the animation")


class ScriptGenerator(curator.LLM):
    """Generates complete animation scripts for mathematical questions"""

    response_format = Script

    def prompt(self, input: Dict) -> str:
        return f"""
        Create a complete animation script for a short (2-3 minute) mathematical explainer video in the style of 3Blue1Brown.
        
        SUBJECT: {input['subject']}
        TOPIC: {input['topic']}
        QUESTION: {input['question']}
        INSIGHT: {input['brief_insight']}
        VISUALIZATION POTENTIAL: {input['visualization_potential']}
        
        Your script should:
        1. Introduce the question in an engaging way
        2. Build intuition step by step
        3. Reveal the key insight
        4. Include clear instructions for visual elements
        5. Specify any equations that should be displayed
        
        The script should be written for narration (what would be spoken) and include notes for visual elements
        that would appear at specific moments. Focus on clarity, building intuition, and revealing mathematical beauty.
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "title": "An engaging title for this animation",
            "narration": "The complete narration script with paragraph breaks...",
            "visual_elements": [
                {{
                    "timestamp": "0:00-0:15",
                    "description": "Description of what should be shown visually"
                }},
                // Additional visual elements
            ],
            "equations": [
                "Equation 1 in LaTeX format",
                "Equation 2 in LaTeX format"
            ],
            "key_timestamps": {{
                "Introduction": "0:00",
                "Key concept 1": "0:30",
                "Key concept 2": "1:00",
                "Key concept 3": "1:30",
                "Key Insight": "2:00",
                "Conclusion": "2:30"
            }},
            "visual_style": "Brief description of the overall visual style and color palette"
        }}
        """

    def parse(self, input: Dict, response: Script) -> Dict:
        return {
            "subject": input["subject"],
            "topic": input["topic"],
            "question": input["question"],
            "title": response.title,
            "narration": response.narration,
            "visual_elements": response.visual_elements,
            "equations": response.equations,
            "key_timestamps": response.key_timestamps,
            "visual_style": response.visual_style,
        }


# Define generators for each level of content
class SubjectGenerator(curator.LLM):
    """Generates diverse mathematical subjects"""

    response_format = Subjects

    def prompt(self, input: Dict) -> str:
        return f"""
        Generate {input['num_subjects']} specific mathematical subjects that have rich potential for exploration.
        
        Your task is to identify a clear, well-defined area of mathematics that contains many interesting topics and questions.
        
        CONSTRAINTS:
        - The subject should be a recognized field or subfield of mathematics
        - It should have connections to multiple other areas of mathematics or science
        - It should offer rich visual and conceptual possibilities
        
        Provide a short description that explains what the subject encompasses.
        
        Choose from the following subjects first and then add similar subjects:
        - Number Theory
        - Topology
        - Differential Geometry
        - Graph Theory
        - Complex Analysis
        - Abstract Algebra
        - Probability Theory
        - Combinatorics
        - Dynamical Systems
        - Game Theory
        - Numerical Analysis
        - Mathematical Physics
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "subjects": [
                {{
                    "index": 0,
                    "name": "The name of the mathematical subject",
                    "description": "A concise description of what this subject encompasses"
                }},
                ...
            ]
        }}
        """

    def parse(self, input: Dict, response: Subjects) -> Dict:
        return [{"subject_name": subject.name, "subject_description": subject.description, "subject_index": subject.index} for subject in response.subjects]


class TopicGenerator(curator.LLM):
    """Generates topics within a mathematical subject"""

    response_format = Topics

    def prompt(self, input: Dict) -> str:
        return f"""
        Generate {input['num_topics']} specific mathematical topics within the subject of {input['subject_name']}.
        
        Your task is to identify {input['num_topics']} focused, interesting topics within this subject that would be rich for exploration.
        
        SUBJECT: {input['subject_name']}
        DESCRIPTION: {input['subject_description']}
        
        CONSTRAINTS:
        - The topic should be specific (not too broad)
        - It should have strong visual potential (can be illustrated effectively)
        - It should offer opportunities for mathematical insight
        
        EXAMPLES OF GOOD SPECIFIC TOPICS:
        - For Number Theory: "Primality Testing Algorithms" or "Distribution of Prime Numbers"
        - For Geometry: "Projective Transformations" or "Curvature of Surfaces"
        - For Probability: "Random Walks" or "Markov Chains"
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "topics": [
                {{
                    "index": 0,
                    "name": "The name of the mathematical topic",
                    "description": "A concise description of what this topic covers",
                    "subject": "{input['subject_name']}"
                }},
                ...
            ]
        }}
        """

    def parse(self, input: Dict, response: Topics) -> Dict:
        return [
            {
                "subject_name": input["subject_name"],
                "subject_description": input["subject_description"],
                "topic_name": topic.name,
                "topic_description": topic.description,
                "topic_index": topic.index,
            }
            for topic in response.topics
        ]


class QuestionGenerator(curator.LLM):
    """Generates interesting questions about mathematical topics"""

    response_format = Questions

    def prompt(self, input: Dict) -> str:
        return f"""
        Generate {input['num_questions']} interesting, specific mathematical questions related to the topic of {input['topic_name']}.
        
        SUBJECT: {input['subject_name']}
        TOPIC: {input['topic_name']}
        TOPIC DESCRIPTION: {input['topic_description']}
        
        Your task is to create {input['num_questions']} thought-provoking mathematical questions that:
        1. Is specific and clearly defined
        2. Has visual potential (can be illustrated well)
        3. Challenges the viewer to think in a new way
        
        EXAMPLES OF GOOD QUESTIONS:
        - "How can we visualize the infinitude of prime numbers?"
        - "Why does multiplying complex numbers rotate the complex plane?"
        - "What happens when we iterate simple functions repeatedly?"
        - "How do random walks converge to smooth curves?"
        - "Why is the determinant equal to the volume of a parallelepiped?"
        
        YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT (JSON):
        {{
            "questions": [
                {{
                    "index": 0,
                    "question": "A clear, specific mathematical question",
                    "brief_insight": "A brief hint at the answer or approach (1-2 sentences)",
                    "visualization_potential": "How this could be effectively visualized",
                    "subject": "{input['subject_name']}",
                    "topic": "{input['topic_name']}"
                }},
                ...
            ]
        }}
        """

    def parse(self, input: Dict, response: Questions) -> Dict:
        return [
            {
                "subject": question.subject,
                "topic": question.topic,
                "question": question.question,
                "brief_insight": question.brief_insight,
                "visualization_potential": question.visualization_potential,
            }
            for question in response.questions
        ]


def generate_hierarchical_math_content(num_subjects=20, topics_per_subject=5, questions_per_topic=3, output_dataset_name="pimpalgaonkar/math_scripts"):
    """Generate a hierarchical dataset of mathematical content

    Args:
        num_subjects: Number of mathematical subjects to generate
        topics_per_subject: Number of topics to generate per subject
        questions_per_topic: Number of questions to generate per topic
        output_dir: Directory to save the dataset

    Returns:
        Path to the generated dataset
    """
    logger.info(f"Starting generation of {num_subjects} subjects with {topics_per_subject} topics each and {questions_per_topic} questions per topic")

    # Initialize generators
    subject_generator = SubjectGenerator(model_name="gpt-4o")
    topic_generator = TopicGenerator(model_name="gpt-4o")
    question_generator = QuestionGenerator(model_name="gpt-4o")
    script_generator = ScriptGenerator(model_name="gpt-4o")

    # Step 1: Generate list of subjects
    subjects_data = Dataset.from_list([{"num_subjects": num_subjects}])
    subjects_dataset = subject_generator(subjects_data)

    # Step 2: Generate topics for each subject
    subjects_dataset = subjects_dataset.add_column("num_topics", [topics_per_subject] * len(subjects_dataset))
    topics_dataset = topic_generator(subjects_dataset)

    # Step 3: Generate questions for each topic
    topics_dataset = topics_dataset.add_column("num_questions", [questions_per_topic] * len(topics_dataset))
    questions_dataset = question_generator(topics_dataset)

    # Step 4: Generate scripts for each question
    scripts_dataset = script_generator(questions_dataset)

    # Step 5: Save the final dataset
    # push to hub
    scripts_dataset.push_to_hub(output_dataset_name, private=True)

    # Return the path to the dataset
    return dataset_path


def analyze_hierarchical_dataset(dataset_path, output_dir="."):
    """Analyze the structure and diversity of the hierarchical dataset

    Args:
        dataset_path: Path to the dataset
        output_dir: Directory to save the analysis

    Returns:
        Dictionary with analysis results
    """
    # Load dataset
    dataset = load_dataset(dataset_path)

    # Analyze subject distribution
    subjects = {}
    for item in dataset:
        subject = item["subject"]
        if subject in subjects:
            subjects[subject] += 1
        else:
            subjects[subject] = 1

    # Analyze topic distribution
    topics = {}
    for item in dataset:
        topic = item["topic"]
        if topic in topics:
            topics[topic] += 1
        else:
            topics[topic] = 1

    # Count unique subjects and topics
    unique_subjects = len(subjects)
    unique_topics = len(topics)

    # Sort results
    subjects = dict(sorted(subjects.items(), key=lambda x: x[1], reverse=True))
    topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:30])

    # Compile analysis
    analysis = {
        "total_questions": len(dataset),
        "unique_subjects": unique_subjects,
        "unique_topics": unique_topics,
        "subject_distribution": subjects,
        "top_topics": topics,
    }

    # Save analysis
    analysis_file = os.path.join(output_dir, "dataset_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Dataset analysis saved to {analysis_file}")

    return analysis


import argparse

if __name__ == "__main__":
    # For small test run, use a smaller number
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_subjects", type=int, default=20)
    parser.add_argument("--topics_per_subject", type=int, default=5)
    parser.add_argument("--questions_per_topic", type=int, default=3)
    parser.add_argument("--output_dataset_name", type=str, default="pimpalgaonkar/math_scripts")
    args = parser.parse_args()

    # Generate the dataset
    dataset_path = generate_hierarchical_math_content(
        num_subjects=args.num_subjects,
        topics_per_subject=args.topics_per_subject,
        questions_per_topic=args.questions_per_topic,
        output_dir="math_hierarchical_content",
        output_dataset_name=args.output_dataset_name,
    )

    # Analyze the dataset
    analysis = analyze_hierarchical_dataset(args.output_dataset_name)

    print("\nDataset generation complete!")
    print(f"Total questions: {analysis['total_questions']}")
    print(f"Unique subjects: {analysis['unique_subjects']}")
    print(f"Unique topics: {analysis['unique_topics']}")

    top_subjects = list(analysis["subject_distribution"].items())[:5]
    print("\nTop 5 subjects:")
    for subject, count in top_subjects:
        print(f"- {subject}: {count} questions ({count/analysis['total_questions']*100:.1f}%)")
