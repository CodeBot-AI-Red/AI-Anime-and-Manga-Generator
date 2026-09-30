# AI Anime and Manga Generator

Estrutura inicial para uma futura IA profissional especializada em geração de imagens de anime, cenas estáticas de anime, arte para mangá, consistência de personagens, poses e enquadramentos e cenários, com suporte futuro para páginas completas de mangá e sequências de cenas.

## Base de difusão local

O projeto inclui um denoiser PyTorch treinado do zero, sem pesos ou APIs externas.
O treino usa *classifier-free guidance* (descarta o condicionamento textual em
parte dos lotes) e a geração combina as previsões condicional e incondicional.
Isso torna a aderência ao prompt ajustável por `guidance_scale` (padrão `3.0`)
na chamada `DiffusionTrainingModel.sample`. O sampler também aceita poucos
passos de inferência corretamente, usando transições DDIM entre os timesteps
selecionados em vez de aplicar uma posterior DDPM de um passo em saltos longos.
