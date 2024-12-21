## Intro to LLMs

LLMs are just probabilistic models which compute a **probability distribution** for the following word in a given context or sentence. The parameters of the model, which describe the distribution of words that can get generated, can be changed through either **prompting or training**. Decoding is the term for generating text with an LLM.

An **embedding** is a numerical representation of text, in a vector of numbers, which tries to capture the word and its **semantic meaning**. In order to create an embedding we use **encoders**. The opposite, to convert to the original word, is called a **decoder**. 

*Attention is all you need* is the paper that made a breakthrough with the transformer architecture.

- **Encoders** take a sequence of tokens and output their numerical representation. Can be used for vector search of text, where each is converted to a numerical representation and then compared. Examples are **BERT**, MiniLM, SBER, etc.
- **Decoders** takes a sequence of tokens and outputs the following token at a time, but only one token. To generate more, we would append the generated tokens to the input in order to keep producing tokens. Examples are **GPT-4**, Llama, Bloom, etc.
- **Encoder-Decoders** are useful for translation of text, since we can compare the semantic meaning between two languages at the vector phase, and then also generate text in the target language. e.g.: **BART**

General standard for models depending on the use case:

![](assets/Pasted%20image%2020240602183118.png)

### Prompting and Training

**Prompt engineering** is refining a prompt (the input) of a model to give results that are closer to the desired output. Because we **cannot see the probability distribution behind the model** and the next word to be generated, the only way to perform this is by **changing the prompt multiple times** until it matches our desired output. To make prompts more effective and repeatable, there are some prompt design patterns and techniques.

- We can divide prompting a model into 3 sections: **task description**, **examples** and **prompt**. By giving out examples, called **in-context learning**, we give the model context on how to perform the task. An example of how we do this is with *k-shot prompting*.
- We can break down the problem into a **Chain-of-thought** in order to make it more manageable.

A risk with prompting is prompt injection. We should never let users have access to the direct input to the model, but only to an input that has implemented mechanisms of control. For example, we wouldn't like our re-trained model to output customer information when it is requested.

**Training** a model is performed to give it a domain adaptation in order to enhance its performance. This is done using labeled datasets. 

- **Fine-tuning** is the most expensive way to train a model because it tries to change **all of its parameters**. If in 2019 this was already expensive, imagine now with models of 100B+ params.
- **Parameter efficient FT** is cheaper because it **only changes some of the parameters**, or instead it just adds new ones without modifying the original ones, such as with the **LORA** method.
- **Soft prompting** is passing **parameters through the prompt**.
- **Continuous pre-training** is an unsupervised task and it is **how the models are originally trained**. So, continuing this process with even more domain-specific data should work but keep changing all the parameters as well.

![](assets/Pasted%20image%2020240602222812.png)

### Decoding

**Greedy** gets the highest probability word from the probability distribution

**Nucleus sampling** adds non-determinism to the output by giving the distribution a **random** component in order to choose different words. It uses the **temperature** hyper parameter, a cold (low) temperature reduces luck, and increasing it increases probability distribution. Also, we can modify what we sample with **top-k** or **top-p** in order to delimit our final words.

[**Beam sampling**](https://www.geeksforgeeks.org/introduction-to-beam-search-algorithm/): it is a bit of a mix of both above. Instead of just generating the next word, it **generates the entire sentence, but multiple times simultaneously**. With the non-deterministic generated sentences we select the one with the overall higher probability and continue from there, repeating the process.

When we generate information from an LLM, in a certain way we are always only getting the most likely words, like a prediction of what follows. We call **hallucinations** to LLM outputs which state something as factual something that is not in real life. Can be very hard to spot. This makes LLMs risky to deploy into a product.

In order to check this phenomenon and avoid it, **attributability** is a common subject of research. This concept means that whenever LLMs generate content, it should be **grounded** in (supported by) a document (like citations).

### Applications of LLMs

Retrieval Augmented Generation (**RAG**) is a way to decrease hallucinations and increase attributability and information that is grounded in documents. Before answering a prompt, we check for relevant information from a data source and then, based on it, we generate a grounded answer.

- **Retrieval**: because we get custom, additional data from a vector database
- **Augmented**: because we augment the original answer with the sourced information 
- **Generation**: generating relevant tokens with said data

![](assets/Pasted%20image%2020240602231113.png)

[**Language agents**](https://yusu.substack.com/p/language-agents#%C2%A7language-agents-a-conceptual-framework) are more than just models. These are systems capable to create plans and reason through a problem, take action through tools, directly impacting their surrounding environment, and go back to planning a next move, over and over again. Agents only stop until stopped by someone or when it concludes it has performed the given task. 

Agents could be very powerful if we can figure out how to give them the tools they should use and a clear objective of what we want. Examples of advancements in this area are Auto-GPT, ReAct, Toolformer, etc.

## Overview of Generative AI in OCI

Four key takeaways:

- Fully managed service, just like any other cloud service.
- They have Cohere and Llama models available to use
- Available tasks are **generation**, **summarization**, **embedding** (for multilingual models)
- Clusters of GPUs, where a clients GPUs are isolated from other users

We can see an **Overview** of the current clusters, endpoints and training jobs. We can create a new cluster in order to either fine-tune or host a model. Fine-tuning a model is done with the T-Few algorithm (cheaper) or with a full fine-tuning process (expensive) And whenever we have a hosted model, we can create an endpoint for it.

Something really cool is that you can play round with a model in the **Playground** area and after doing this you can export your prompt, hyper parameter settings, etc. into a Python code which uses the `oci` module to perform calles directly from their SDK.

## Generation Models

First, a very powerful model is the **generation model**.

Two properties of every model is their **parameter size**, which can vary a lot, and its **context window**, which is defined as the amount of tokens in $\text{input}+\text{output}$

Hyperparameters that can be changed in the **Playground** window:

- A **stop sequence** is a string or token which, when generated, will make the model stop generating more text. For example, a stop sequence of `.` would generate a single sentence.
- The **presence/frequency penalty** is used to avoid repeating tokens and repetitive text, by performing a penalty on the probability of tokens that have appeared multiple times.
- The **show likelihood** is basically a way to describe the probability distribution in a continuous range of floating point numbers of \[-15, 0\].

You can also export your current configuration to code, which performs the same calls in the Playground but through the **Inference API** on the `oci` Python module. In order to start calling the API a profile config file must first be correctly setup. Inside the config file is saved the user, profile, fingerprint, API key reference, etc.

With the profile config setup, the main takeaways from the exported code are:

- Correctly import the profile config file and specify auth config in code
- Define serving endpoint for the compartment to be used. Compartments are similar to GCP projects but on OCI they can be nested and stuff.
- Define the inference request details such as prompt, hyperparameters, model to use, etc.
- Join all settings above into the correct OCI SDK objects
- Finally, call `generate_text(generate_text_details)` and obtain back a dictionary (JSON).
- Inside the response is some metadata and the `text`

Apart from the generation model, **summarization models** are useful to make a text more concise. It has hyperparameters for temperature, length and extractiveness, which defines how much to summarize, if verbatim (with the same words) or paraphrasing.

## Embedding Models

Embeddings models use encoders to represent words numerically. 

Each of the numbers in the vector representation of a word is a **semantic property** of said word. For example, one of these properties could be from the word "dog", which probably has numbers that represent its age, size, etc. 

We can check the **semantic similarity** of two words through **embedding similarities**, which can be performed with cosine and dot product similarity techniques across different dimensions (across different properties). We can even compare sentences between each other to check similarity, and might find out that, numerically, `sound cat makes` $\approx$ `meow`.

How RAG systems work:

![](assets/Pasted%20image%2020240605150915.png)

Their use cases also involve semantic search, text clustering and classification.

In OCI there's Cohere's `Embed-Multilingual` and `Embed-Multilingual-Light`. Models in general will have a set number of dimensions that they will generate per embedding and a maximum number of tokens to consider. So $\text{dimensions}\times\text{tokens}$ will be the embedding size.