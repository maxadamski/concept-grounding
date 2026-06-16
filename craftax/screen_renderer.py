import jax
import jax.numpy as jnp
import numpy as np

from craftax import render_craftax_pixels, OBS_DIM, BLOCK_PIXEL_SIZE_HUMAN, INVENTORY_OBS_HEIGHT

class Renderer:
    def __init__(self, upscale=4):
        self.upscale = upscale
        self.screen_shape = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * upscale,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * upscale,
        )
        self.render_jit = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def render(self, env_state):
        x = self.render_jit(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN).transpose((1, 0, 2))
        x = jnp.repeat(x, repeats=self.upscale, axis=0)
        x = jnp.repeat(x, repeats=self.upscale, axis=1)
        x = np.array(x, dtype=np.uint8)
        return x
