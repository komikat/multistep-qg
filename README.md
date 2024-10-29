### Multi step Question generation

#### Baselines
##### Zero Shot prompted llama3.2-1B Instruct
We have implemented a baseline for the multi hop question generation task.
For this baseline, we use the Llama3.2-1B model as the generator.

We used BLEU score and ROUGE score as metrics for computing similarity between the generated question and ground truth question (provided in the dataset).

- BLEU     = `0.0043`
- ROUGE-1  = `0.1383`
- ROUGE-2  = `0.0210`
- ROUGE-L  = `0.1045`

Here is an example of the generated question, given the context and answer:

Context:
```
El-P (Jaime Meline) is a visionary force in underground hip hop, known not only for his groundbreaking music but also for shaping the cultural framework of the genre. As a producer, rapper, and co-founder of the influential label Definitive Jux, El-P has worked with some of the most celebrated names in alternative hip hop, including Aesop Rock, Mr. Lif, and Cage. His production style blends gritty, experimental beats with socially conscious lyrics, pushing the boundaries of traditional hip hop. Beyond the music, El-P has contributed to the deeper understanding of hip hop culture, coining the idea of nine distinct elements, which go beyond the standard four (MCing, DJing, graffiti, and breakdancing). With projects ranging from solo work to his collaborations in the acclaimed duo Run the Jewels, El-P remains a pivotal figure in defining what hip hop can be, both musically and culturally.
```

Answer:
```
Jaime Meline
```

Generated Question:
```
Which American hip hop recording artist, record producer, and record executive is known for coining the terms of the nine distinct elements of hip hop culture and is credited with producing music for notable rappers such as Aesop Rock, Mr. Lif, and Cage?
```

Next, we plan on implementing multi-hop-question-generation by retrieving useful entities and relationships from knowledge graphs created from the context.