from bespokelabs.curator.blocks.raft import Raft

TEXT = """

The history of human civilization spans thousands of years, with early societies emerging around major rivers such as the Nile, Tigris, Euphrates, Indus, and Yellow Rivers. The first known civilization, Sumer, flourished in Mesopotamia around 3100 BCE. The Sumerians developed the first writing system, cuneiform, and built ziggurats, which served as religious and administrative centers.

Meanwhile, in ancient Egypt, the civilization along the Nile thrived for over 3,000 years. The Egyptians developed a complex society with a centralized monarchy led by pharaohs, who were believed to be divine. They built monumental pyramids, mastered irrigation, and created a sophisticated system of hieroglyphic writing.

The Indus Valley Civilization (c. 2600–1900 BCE) developed in present-day Pakistan and India, featuring advanced urban planning, drainage systems, and standardized weights and measures. Unlike Mesopotamia and Egypt, little is known about their rulers or governance, as their script remains undeciphered.

In China, the Shang Dynasty (c. 1600–1046 BCE) was the first recorded ruling dynasty, known for its oracle bone inscriptions and bronze craftsmanship. The Zhou Dynasty followed, introducing the concept of the Mandate of Heaven, which justified the rule of emperors based on divine approval.

Across the Atlantic, the Olmec civilization (c. 1200–400 BCE) emerged in Mesoamerica, known for its colossal stone heads and influence on later cultures like the Maya and Aztecs. The Maya (c. 250–900 CE) developed a complex calendar

"""


raft = Raft(model="gpt-4o-mini", distractors=3, chunk_size=100, n_questions=3, p=0.8)
dataset = raft(TEXT)
