# Roadmap

## Etapa 7 — fundação de difusão (concluída)

* Denoiser U-Net compacto, inicializado do zero, para predição de ruído em imagens locais 32×32.
* Scheduler DDPM linear, condicionamento de prompt byte-level e checkpoints retomáveis.
* Estrutura pronta para 64×64 e 128×128 sem reescrever a rede.

## Próximo estágio recomendado

Treinar e avaliar a fundação com um dataset local anotado, então acrescentar blocos U-Net adicionais, encoder de texto melhor treinado do zero, augmentação local e métricas/visualização. Apenas após isso considerar autoencoder latente, condicionamentos de pose/câmera e resoluções maiores. O projeto ainda não busca qualidade profissional, consistência avançada de personagem ou páginas de mangá completas.
