# Project Limitations

## Evaluation scope

The API experiment used a fixed set of 800 records, not all 14,755 processed records. The results describe this pilot set and should not be treated as full-dataset performance.

## Dataset differences

The four datasets differ in task type, language, and answer format. Because of this, the overall accuracy is a combined summary rather than the score of one uniform benchmark.

## Model and prompt configuration

The final comparison used the prompt that performed better for each model in the A/B experiment: Prompt B for DeepSeek V4 Pro and Prompt A for Solar Pro 3. The results therefore compare two model-and-prompt configurations, not the models alone under identical prompts.

## Automatic scoring

Automatic scoring can be affected by response format, numerical precision, and different but equivalent ways of expressing an answer. The evaluation rules handle the expected formats, including displayed-precision tolerance and percent/decimal equivalence for FinQA, but they may not capture every valid variation.

## API conditions

Latency, estimated cost, and model outputs reflect the APIs, model versions, and network environment used when the experiment was run. These values or responses may differ under other conditions.
