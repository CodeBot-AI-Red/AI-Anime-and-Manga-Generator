# Arquitetura do primeiro modelo generativo experimental

## Fluxo de geração

O primeiro modelo generativo local implementa o fluxo `Prompt -> Text
Conditioning -> Representação latente -> Modelo generativo -> Geração
iterativa -> Imagem`. Ele é pequeno, determinístico e usa apenas Python puro:
serve para validar interfaces e execução local, não para qualidade visual
profissional.

1. `TextConditioner` transforma cada prompt em um embedding com dimensão
   configurável. A interface aceita lotes e permanece isolada para a futura
   troca por tokenizer e encoder de texto treinável.
2. `LatentRepresentation` reduz imagens BCHW por blocos espaciais e repete os
   valores no decoder. Ela define o contrato encoder/decoder entre dataset,
   treinamento e geração.
3. `SmallGenerativeModel` recebe o mapa latente, embedding textual e índice da
   etapa. O embedding seleciona uma correção por canal e a etapa controla sua
   intensidade.
4. `IterativeImageGenerator` cria um latente inicial determinístico do prompt,
   aplica a correção por `generation_steps` e decodifica o latente para uma
   imagem normalizada `(1, C, H, W)`.

## Treinamento e configurações

`GenerativeTrainingModel` usa o `ProcessedDataLoader` e o `Trainer` já
existentes. Ele codifica imagens pré-processadas, aplica uma etapa condicionada
por suas captions e decodifica a saída; portanto não cria dataset ou loop de
treinamento paralelo. O pequeno `scale` e `bias` treináveis apenas validam o
contrato de otimização nesta etapa.

`ModelConfig` e `configs/model.yaml` expõem `image_resolution`,
`latent_channels`, `downsample_factor` e `generation_steps`. Os valores são
deliberadamente modestos e não há download de pesos, APIs externas ou arquivos
de pesos versionados.

## Evolução profissional

No futuro, o condicionador determinístico será substituído por tokenizer,
encoder textual e atenção cross-modal. A representação latente poderá se tornar
um VAE treinável; o modelo pequeno, uma U-Net ou transformer de difusão com
predição de ruído; e a atualização simples, um scheduler de difusão. Controles
de personagem, pose, enquadramento e estilo da estrutura de cena poderão entrar
como condicionamentos adicionais. Essas trocas preservam as APIs de latente,
condicionamento e geração iterativa, permitindo crescer para um modelo
especializado de anime e mangá sem reescrever a organização do projeto.
