# Arquitetura de difusão treinável

## Fluxo de geração

O caminho treinável é `imagem limpa -> ruído/timestep -> U-Net condicional -> ruído previsto`. Ele usa PyTorch somente como backend local de autograd e inicializa todos os pesos do zero: não carrega pesos, modelos, datasets ou APIs externas.

`DiffusionDenoiser` trabalha diretamente em imagem BCHW e usa uma U-Net de dois níveis: blocos residuais FiLM em cada resolução, duas reduções espaciais, conexões de *skip* e blocos de reconstrução simétricos. A atenção cruzada no gargalo permite que cada região da imagem escolha tokens relevantes do prompt, em vez de receber somente um vetor médio global. A resolução precisa ser divisível por quatro; portanto, os valores padrão 32×32, 64×64 e 128×128 são suportados.

Cada timestep é convertido por `TimestepEmbedding` sinusoidal e projeções treináveis. `PromptEncoder` é um Transformer byte-level treinável do zero, com embeddings posicionais e máscara de padding. Seus tokens contextualizados alimentam tanto o vetor global que modula os blocos residuais por escala e viés como a atenção cruzada no gargalo. Durante o treino, o *classifier-free guidance* descarta a caption de forma independente por amostra, preservando uma rota incondicional sem duplicar o modelo.

`NoiseScheduler` cria `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise`; a MSE compara a saída do denoiser com esse `noise`. No sampling, começa com ruído gaussiano e usa transições DDIM corretas inclusive quando poucos passos de inferência saltam timesteps.

## Treino, capacidade e checkpoints

`DiffusionTrainingModel` adapta o `ProcessedDataLoader` e `Trainer` existentes, portanto usa imagens e metadados já processados sem segundo dataset. Seus checkpoints `.pt` incluem pesos, configuração e estado AdamW para retomada.

Os valores padrão atuais (`base_channels=16`, condicionamento de 64 dimensões, dois blocos por nível e encoder textual de duas camadas) totalizam **542.515 parâmetros treináveis**, contra **27.947** na U-Net anterior: **514.568** parâmetros adicionais. O crescimento vem de capacidade funcional (profundidade multiescala, encoder sequencial e atenção texto-imagem), não de tensores sem uso. `count_trainable_parameters` expõe a contagem exata, que o script de treino também imprime.

## Limitações e evolução

Esta ainda é uma fundação pequena, treinada somente com dados locais, e não produz anime profissional sem um dataset curado, resolução maior e tempo de treino suficiente. O condicionamento de cena/pose/identidade permanece como interface estrutural fora do denoiser; controles espaciais explícitos e consistência entre quadros ainda não foram implementados.

Próximos passos coerentes são um autoencoder latente treinado do zero para escalar resolução, condicionadores estruturados de pose/layout/identidade ligados à atenção cruzada, augmentação local, EMA dos pesos para sampling, predição `v` e avaliação visual/reconstrução em conjunto de validação local. Essas trocas preservam scheduler, checkpoints e o loop de treino atuais.
