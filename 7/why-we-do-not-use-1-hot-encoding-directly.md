# Why Not Feed the One-Hot Vector Directly into the RNN?

When working with natural language processing (NLP), a common question arises:

**If we can represent every word as a one-hot vector, why do we need an embedding layer before feeding the words into an RNN?**

After all, a one-hot vector already uniquely identifies a word. Why add another step?

To understand this, we need to look at what one-hot vectors represent, what an RNN needs, and why embeddings are a better input representation.

---

## 1. What Is a One-Hot Vector?

A one-hot vector is a way of representing a categorical value using a vector containing only zeros and one single `1`.

For example, suppose our vocabulary contains five words:

| Index | Word |
|---|---|
| 0 | hello |
| 1 | world |
| 2 | I |
| 3 | love |
| 4 | AI |

The one-hot representations would be:

```
hello → [1, 0, 0, 0, 0]

world → [0, 1, 0, 0, 0]

I     → [0, 0, 1, 0, 0]

love  → [0, 0, 0, 1, 0]

AI    → [0, 0, 0, 0, 1]
```

The position of the `1` tells us which word it represents.

A vocabulary of 50,000 words would require a vector of length 50,000 for every word.

---

# 2. Why Not Feed These Vectors Directly into an RNN?

Technically, we can.

An RNN accepts vectors as input, so there is no mathematical problem with feeding a one-hot vector directly.

For example:

```
Word:
"love"

One-hot representation:

[0,0,0,1,0]
```

This vector can be passed into an RNN.

However, there are several problems.

---

# Problem 1: One-Hot Vectors Are Extremely Large

Real-world vocabularies are huge.

Suppose:

```
Vocabulary size = 100,000 words
```

Then every word becomes:

```
[0,0,0,0,0,...,1,...,0]
```

with 100,000 dimensions.

Imagine a sentence:

```
"I love machine learning"
```

Each word would be a vector of size 100,000.

The RNN would have to process enormous sparse vectors.

Most of the values are useless zeros.

---

# Problem 2: One-Hot Vectors Have No Understanding of Meaning

One-hot encoding only tells the model:

> "This is word number 523."

It does not tell the model anything about the word itself.

For example:

```
king  → [1,0,0,0]

queen → [0,1,0,0]

car   → [0,0,1,0]

truck → [0,0,0,1]
```

From these vectors, the model cannot know:

```
king is similar to queen

car is similar to truck
```

Every word is equally unrelated to every other word.

The distance between:

```
king and queen
```

is the same as:

```
king and banana
```

because all one-hot vectors are orthogonal.

---

# Problem 3: The RNN Has to Learn a Huge Transformation

An RNN does not directly understand the meaning of the input vector.

It applies transformations using weight matrices.

Suppose:

```
Vocabulary size = 100,000

RNN hidden size = 256
```

The RNN needs to transform:

```
100,000-dimensional input
```

into:

```
256-dimensional hidden representation
```

This requires a huge weight matrix:

```
100,000 × 256
```

That is approximately:

```
25.6 million parameters
```

just to convert words into useful representations.

This is inefficient.

---

# 3. What Does an Embedding Layer Do Instead?

An embedding layer replaces the huge one-hot vector with a smaller dense vector.

Instead of:

```
word
 |
 v
100,000-dimensional one-hot vector
 |
 v
RNN
```

we do:

```
word index
 |
 v
Embedding layer
 |
 v
256-dimensional vector
 |
 v
RNN
```

For example:

```
king:

[0.21, 0.52, 0.18, 0.76]


queen:

[0.20, 0.49, 0.22, 0.71]
```

Now the vectors contain learned information.

The model can learn that:

```
king ≈ queen

dog ≈ cat

car ≈ truck
```

because their embeddings become close together.

---

# 4. Is an Embedding Just a Faster One-Hot Multiplication?

Yes.

Mathematically:

```
one-hot vector × embedding matrix
```

produces the same result as:

```
embedding lookup
```

Suppose:

```
Vocabulary:

0: cat
1: dog
2: car
```

The embedding matrix might be:

```
        dimension
          ↓

cat   [0.2, 0.5, 0.1]

dog   [0.3, 0.6, 0.2]

car   [0.8, 0.1, 0.7]
```

The one-hot vector:

```
dog

[0,1,0]
```

multiplied by the matrix selects:

```
[0.3,0.6,0.2]
```

The embedding layer simply performs this lookup directly without creating the large one-hot vector.

---

# 5. Why Does the Embedding Have Trainable Values?

Initially, embeddings are usually random.

Example:

```
dog:

[0.13, -0.42, 0.91]


cat:

[-0.22, 0.51, 0.34]
```

During training, the neural network adjusts these values.

Words that appear in similar contexts move closer together.

For example:

Sentences:

```
The dog chased the ball.

The cat chased the mouse.
```

The model learns that:

```
dog ≈ cat
```

because they behave similarly in language.

---

# 6. Why Not Use One-Hot Vectors and Let the RNN Learn Everything?

The RNN could theoretically learn the same representations.

However:

- The input dimension becomes extremely large.
- Training becomes slower.
- The number of parameters increases significantly.
- The model must discover word relationships from scratch.
- Memory usage becomes much higher.

An embedding layer separates the problem:

1. Convert words into meaningful dense vectors.
2. Process those vectors with the RNN.

This makes learning much more efficient.

---

# 7. The Complete Pipeline

The typical NLP pipeline is:

```text
        Sentence:
        "I am Groot"
            |
            v
        Word indices:   
        [12, 45, 891]
            |
            v
    Embedding layer:
    [
    [0.23,0.54,0.12],
    [0.11,0.72,0.43],
    [0.88,0.21,0.34]
    ]
            |
            v
      Encoder-Decoder
            |
            v
        Prediction
```

---

# Conclusion

A one-hot vector is useful because it uniquely identifies a word, but it is not a good representation for learning language.

It is:

- Huge
- Sparse
- Lacking semantic information

An embedding layer provides a compact, dense, and learnable representation of words.

So the reason we do not feed one-hot vectors directly into an RNN is not because it is impossible. It is because embeddings provide a much more efficient and meaningful representation for language.