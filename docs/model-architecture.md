# Arquitetura de difusão treinável (Etapa 7)

## Fluxo de geração

O protótipo iterativo anterior permanece para compatibilidade, mas o caminho
novo e treinável é `imagem limpa -> ruído/timestep -> U-Net condicional ->
ruído previsto`. Ele usa PyTorch somente como backend local de autograd e
inicializa todos os pesos do zero: não carrega pesos, modelos ou APIs externos.

`DiffusionDenoiser` trabalha diretamente em imagem BCHW e contém convolução de
entrada, bloco residual, downsample, bloco central, upsample, conexão de skip e
saída com os mesmos canais espaciais da entrada. `ModelConfig.image_resolution`
aceita 32×32 por padrão e o mesmo encoder pode operar em 64×64 ou 128×128,
desde que a dimensão seja divisível por dois.

Cada timestep é convertido por `TimestepEmbedding` sinusoidal e projeções
treináveis. O `PromptEncoder` é um encoder byte-level treinável do zero; a
combinação de ambos modula blocos residuais por escala e viés. Ele é simples de
propósito e pode ser substituído mais tarde por tokenização e condicionamento
cross-attention sem mudar o scheduler ou o loop de treino.

`NoiseScheduler` tem betas lineares configuráveis. No treino ele constrói
`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise`; a MSE compara a
saída do denoiser com esse `noise`. No sampling, começa com ruído gaussiano e
aplica passos DDPM reversos até gerar uma imagem normalizada.

## Limitações e evolução

`DiffusionTrainingModel` adapta o `ProcessedDataLoader` e `Trainer` existentes,
portanto usa as imagens e metadados já processados sem segundo dataset. Seus
checkpoints `.pt` incluem pesos e estado AdamW para retomada.

`ModelConfig` e os YAMLs expõem resolução, canais-base, parâmetros de
condicionamento, timesteps, passos de sampling, seed e parâmetros de treino.
Os valores são deliberadamente modestos e não há download de pesos, APIs
externas ou arquivos de pesos versionados.

## Evolução profissional

O modelo é uma fundação pequena e não produz anime profissional. Próximos
passos incluem VAE treinável, U-Net mais profunda, melhor encoder de texto
treinado no dataset, mais resolução e validação visual. Controles
de personagem, pose, enquadramento e estilo da estrutura de cena poderão entrar
como condicionamentos adicionais. Essas trocas preservam as APIs de latente,
condicionamento e geração iterativa, permitindo crescer para um modelo
especializado de anime e mangá sem reescrever a organização do projeto.
