import random
import torch


dog_cat_words = ['dog', 'cat', 'pet', 'house', 'animal', 'sleep', 'play']
family_words = ['girl', 'boy', 'father', 'mother', 'family', 'house', 'marriage']
king_queen_words = ['crown', 'queen', 'king', 'empire', 'country', 'rule', 'castle']
unique_words = dog_cat_words+family_words+king_queen_words
"""

dog_cat_text = ''
family_text = ''
king_queen_text = ''

for _ in range(10_000):
    random.shuffle(dog_cat_words)
    dog_cat_text = dog_cat_text + ' ' + ' '.join(dog_cat_words)
    random.shuffle(family_words)
    family_text = family_text + ' ' + ' '.join(family_words)
    random.shuffle(king_queen_words)
    king_queen_text = king_queen_text + ' ' + ' '.join(king_queen_words)

small_corpus = dog_cat_text + ' ' + family_text + ' ' + king_queen_text

# Extract the dataset
file_name = "small_corpus.txt"
with open(file_name, 'w') as file:
    file.write(small_corpus)
"""


# Example usage
file_path = 'small_corpus.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

def word_tokenize(text:str):
    return text.split(" ")

def generate_cbows(text, window_size):
    # Lowercase and tokenize the text
    text = text.lower()
    words = word_tokenize(text)

    # Remove punctuation
    words = [word for word in words if word.isalpha()]

    # Remove stop words
    words = [word for word in words]

    # Create CBOW pairs with a given window size
    cbows = []
    for i, target_word in enumerate(words):
        context_words = words[max(0, i - window_size):i] + words[i + 1:i + window_size + 1]
        if len(context_words) == window_size * 2:
            cbows.append((context_words, target_word))
    return cbows

# Create cbows
cbows = generate_cbows(text, window_size=3)

# Display the results
for context_words, target_word in cbows[:3]:
    print(f'Context Words: {context_words}, Target Word: {target_word}')

def one_hot_encoding(word, unique_words):
    encoding = [1 if word == w else 0 for w in unique_words]
    return torch.tensor(encoding, dtype=torch.float32)

# Create one-hot encodings for each word
one_hot_encodings = {word: one_hot_encoding(word, unique_words) for word in unique_words}

print(one_hot_encodings)

# Convert CBOW pairs to vector pairs
cbow_vector_pairs = [([one_hot_encodings[word] for word in context_words], one_hot_encodings[target_word]) for context_words, target_word in cbows]

print(cbow_vector_pairs)

# Sum the context vectors to get a single context vector
cbow_vector_pairs = [(torch.sum(torch.stack(context_vectors), dim=0), target_vector) for context_vectors, target_vector in cbow_vector_pairs]


print(cbow_vector_pairs)