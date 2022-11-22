<script>
  import Sidebar from "$lib/Sidebar.svelte";
  const gm = [
  {
        name: 'Generative Models',
        link: '/blocks/deep_learning/generative_models',
        links: [
          {
            name: 'Autoregressive Generative Models',
            link: '/blocks/deep_learning/generative_models/autoregressive',
            links: [
              {
                name: 'PixelRNN',
                link: '/blocks/deep_learning/generative_models/autoregressive/pixel_rnn',
              },
              {
                name: 'Gated PixelCNN',
                link: '/blocks/deep_learning/generative_models/autoregressive/gated_pixel_cnn',
              },
              {
                name: 'Implementations',
                link: '/blocks/deep_learning/generative_models/autoregressive/implementations',
                links: [
                  {
                    name: 'RowLSTM',
                    link: '/blocks/deep_learning/generative_models/autoregressive/implementations/row_lstm',
                  },
                  {
                    name: 'PixelCNN',
                    link: '/blocks/deep_learning/generative_models/autoregressive/implementations/pixel_cnn',
                  },
                  {
                    name: 'Gated PixelCNN',
                    link: '/blocks/deep_learning/generative_models/autoregressive/implementations/gated_pixel_cnn',
                  }
                ]
              }
            ]
          },
          {
            name: 'Latent Variable Models',
            link: '/blocks/deep_learning/generative_models/latent_variable_models',
            links: [
              {
                name: 'Normalizing Flows',
                link:   '/blocks/deep_learning/generative_models/latent_variable_models/normalizing_flows',
                links: [
                  {
                    name: 'NICE',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/normalizing_flows/nice',
                  },
                  {
                    name: 'RealNVP',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/normalizing_flows/real_nvp',
                  },
                  {
                    name: 'Glow',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/normalizing_flows/glow',
                  }
                ]
              },
              {
                name: 'Autoencoders',
                link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders',
                links: [
                  {
                    name: 'Variational Autoencoder',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders/variational_autoencoder',
                  },
                  {
                    name: 'VQ-VAE',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders/vq_vae',
                  },
                ]
              },
              {
                name: 'Implementations',
                link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders/implementations',
                links: [
                  {
                    name: 'Autoencoders',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders/implementations/autoencoders',
                  },
                  {
                    name: 'VAE',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders/implementations/vae',
                  },
                  {
                    name: 'VQ-VAE',
                    link: '/blocks/deep_learning/generative_models/latent_variable_models/autoencoders/implementations/vq_vae',
                  }
                ]
              }
            ]
          },
          {
            name: 'Generative Adversarial Networks',
            link: '/blocks/deep_learning/generative_models/generative_adversarial_networks',
          },
          {
            name: 'Diffusion Models',
            link: '/blocks/deep_learning/generative_models/diffusion_models',
            links: [
              {
                name: 'GLIDE',
                link: '/blocks/deep_learning/generative_models/diffusion_models/glide',
              },
              {
                name: 'Latent Diffusion',
                link: '/blocks/deep_learning/generative_models/diffusion_models/latent_diffusion',
              }
            ]
          },
        ]
    }
  ]
</script>

<Sidebar root={gm} />
<main>
  <article>
    <slot />
  </article>
</main>
