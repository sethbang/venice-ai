import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import io
from pathlib import Path

from venice_ai.resources.image import Image, AsyncImage, _guess_image_type
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.types.image import ImageResponse, SimpleImageResponse

class TestImage:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client.post = MagicMock()
        client._request = MagicMock() # Mock the internal _request method
        client.get = MagicMock()
        return client

    def test_generate_with_required_params(self, mock_client):
        """Test generate method with only required parameters."""
        image_resource = Image(mock_client)
        model = "test-model"
        prompt = "test prompt"

        image_resource.generate(model=model, prompt=prompt)

        mock_client.post.assert_called_once_with(
            "image/generate",
            json_data={
                "model": model,
                "prompt": prompt,
            },
            cast_to=ImageResponse
        )
        mock_client._request.assert_not_called() # Ensure _request is not called for non-binary

    def test_generate_with_all_optional_params(self, mock_client):
        """Test generate method with all optional parameters."""
        image_resource = Image(mock_client)
        model = "test-model"
        prompt = "test prompt"
        cfg_scale = 1.5
        embed_exif_metadata = True
        format = "png"
        height = 512
        hide_watermark = False
        lora_strength = 50
        negative_prompt = "ugly"
        return_binary = False
        safe_mode = True
        seed = 123
        steps = 30
        style_preset = "cinematic"
        width = 768

        image_resource.generate(
            model=model,
            prompt=prompt,
            cfg_scale=cfg_scale,
            embed_exif_metadata=embed_exif_metadata,
            format=format,
            height=height,
            hide_watermark=hide_watermark,
            lora_strength=lora_strength,
            negative_prompt=negative_prompt,
            return_binary=return_binary,
            safe_mode=safe_mode,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            width=width
        )

        mock_client.post.assert_called_once_with(
            "image/generate",
            json_data={
                "model": model,
                "prompt": prompt,
                "cfg_scale": cfg_scale,
                "embed_exif_metadata": embed_exif_metadata,
                "format": format,
                "height": height,
                "hide_watermark": hide_watermark,
                "lora_strength": lora_strength,
                "negative_prompt": negative_prompt,
                "return_binary": return_binary,
                "safe_mode": safe_mode,
                "seed": seed,
                "steps": steps,
                "style_preset": style_preset,
                "width": width,
            },
            cast_to=ImageResponse
        )
        mock_client._request.assert_not_called()

    def test_generate_return_binary(self, mock_client):
        """Test generate method when return_binary is True."""
        image_resource = Image(mock_client)
        model = "test-model"
        prompt = "test prompt"
        return_binary = True
        mock_client._request.return_value = b"binary image data"

        result = image_resource.generate(model=model, prompt=prompt, return_binary=return_binary)

        mock_client._request.assert_called_once_with(
            method="POST",
            path="image/generate",
            json_data={
                "model": model,
                "prompt": prompt,
                "return_binary": return_binary,
            },
            headers={"Accept": "image/*"},
            raw_response=True
        )
        mock_client.post.assert_not_called()
        assert result == b"binary image data"

    def test_simple_generate_with_required_params(self, mock_client):
        """Test simple_generate method with only required parameters."""
        image_resource = Image(mock_client)
        model = "test-model"
        prompt = "test prompt"

        image_resource.simple_generate(model=model, prompt=prompt)

        mock_client.post.assert_called_once_with(
            "images/generations",
            json_data={
                "model": model,
                "prompt": prompt,
            },
            cast_to=SimpleImageResponse
        )

    def test_simple_generate_with_all_optional_params(self, mock_client):
        """Test simple_generate method with all optional parameters."""
        image_resource = Image(mock_client)
        model = "test-model"
        prompt = "test prompt"
        background = "transparent"
        moderation = "low"
        n = 5
        output_compression = 80
        output_format = "webp"
        quality = "high"
        response_format = "url"
        size = "512x512"
        style = "vivid"
        user = "test-user"

        image_resource.simple_generate(
            model=model,
            prompt=prompt,
            background=background,
            moderation=moderation,
            n=n,
            output_compression=output_compression,
            output_format=output_format,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
            user=user
        )

        mock_client.post.assert_called_once_with(
            "images/generations",
            json_data={
                "model": model,
                "prompt": prompt,
                "background": background,
                "moderation": moderation,
                "n": n,
                "output_compression": output_compression,
                "output_format": output_format,
                "quality": quality,
                "response_format": response_format,
                "size": size,
                "style": style,
                "user": user,
            },
            cast_to=SimpleImageResponse
        )

    @patch("venice_ai.resources.image.base64.b64encode")
    def test_upscale_with_filepath(self, mock_b64encode, mock_client):
        """Test upscale method with a file path."""
        image_resource = Image(mock_client)
        mock_b64encode.return_value = b"base64encodedimage"
        mock_client._request.return_value = b"upscaled image data"

        # Use the dummy file from test data directory
        dummy_file_path = Path("e2e_tests/data/dummy_image.png")

        result = image_resource.upscale(image=str(dummy_file_path))

        mock_client._request.assert_called_once_with(
            method="POST",
            path="image/upscale",
            json_data={"image": "base64encodedimage", "scale": 2.0},
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_client.post.assert_not_called()
        assert result == b"upscaled image data"

    @patch("venice_ai.resources.image.base64.b64encode")
    def test_upscale_with_bytes(self, mock_b64encode, mock_client):
        """Test upscale method with bytes input."""
        image_resource = Image(mock_client)
        mock_b64encode.return_value = b"base64encodedimage"
        mock_client._request.return_value = b"upscaled image data"

        image_bytes = b"image content bytes"
        result = image_resource.upscale(image=image_bytes)

        mock_client._request.assert_called_once_with(
            method="POST",
            path="image/upscale",
            json_data={"image": "base64encodedimage", "scale": 2.0},
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_client.post.assert_not_called()
        assert result == b"upscaled image data"

    @patch("venice_ai.resources.image.base64.b64encode")
    def test_upscale_with_file_like_object(self, mock_b64encode, mock_client):
        """Test upscale method with a file-like object."""
        image_resource = Image(mock_client)
        mock_b64encode.return_value = b"base64encodedimage"
        mock_client._request.return_value = b"upscaled image data"

        file_obj = io.BytesIO(b"file object content")
        result = image_resource.upscale(image=file_obj)

        mock_client._request.assert_called_once_with(
            method="POST",
            path="image/upscale",
            json_data={"image": "base64encodedimage", "scale": 2.0},
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_client.post.assert_not_called()
        assert result == b"upscaled image data"

    @patch("venice_ai.resources.image.base64.b64encode")
    def test_upscale_with_optional_params(self, mock_b64encode, mock_client):
        """Test upscale method with optional parameters."""
        image_resource = Image(mock_client)
        mock_b64encode.return_value = b"base64encodedimage"
        mock_client._request.return_value = b"upscaled image data"

        image_bytes = b"image content bytes"
        enhance = True
        enhance_creativity = 0.8
        enhance_prompt = True
        replication = 0.5
        scale = 4.0

        result = image_resource.upscale(
            image=image_bytes,
            enhance=enhance,
            enhance_creativity=enhance_creativity,
            enhance_prompt=enhance_prompt,
            replication=replication,
            scale=scale
        )

        mock_client._request.assert_called_once_with(
            method="POST",
            path="image/upscale",
            json_data={
                "image": "base64encodedimage",
                "enhance": True, # True is passed as a boolean
                "replication": replication,
                "scale": scale,
            },
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_client.post.assert_not_called()
        assert result == b"upscaled image data"

    def test_list_styles(self, mock_client):
        """Test list_styles method."""
        image_resource = Image(mock_client)
        mock_client.get.return_value = {"data": [{"id": "style1", "name": "Style 1"}]}

        result = image_resource.list_styles()

        mock_client.get.assert_called_once_with("image/styles")
        assert result == {"data": [{"id": "style1", "name": "Style 1"}]}

class TestAsyncImage:
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client.post = AsyncMock()
        client._request = AsyncMock() # Mock the internal _request method
        client.get = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_generate_with_required_params(self, mock_async_client):
        """Test async generate method with only required parameters."""
        image_resource = AsyncImage(mock_async_client)
        model = "test-model-async"
        prompt = "test prompt async"

        await image_resource.generate(model=model, prompt=prompt)

        mock_async_client.post.assert_awaited_once_with(
            "image/generate",
            json_data={
                "model": model,
                "prompt": prompt,
            },
            cast_to=ImageResponse
        )
        mock_async_client._request.assert_not_called() # Ensure _request is not called for non-binary

    @pytest.mark.asyncio
    async def test_generate_with_all_optional_params(self, mock_async_client):
        """Test async generate method with all optional parameters."""
        image_resource = AsyncImage(mock_async_client)
        model = "test-model-async"
        prompt = "test prompt async"
        cfg_scale = 2.0
        embed_exif_metadata = False
        format = "jpeg"
        height = 1024
        hide_watermark = True
        lora_strength = 75
        negative_prompt = "blurry"
        return_binary = False
        safe_mode = False
        seed = 456
        steps = 40
        style_preset = "photographic"
        width = 1024

        await image_resource.generate(
            model=model,
            prompt=prompt,
            cfg_scale=cfg_scale,
            embed_exif_metadata=embed_exif_metadata,
            format=format,
            height=height,
            hide_watermark=hide_watermark,
            lora_strength=lora_strength,
            negative_prompt=negative_prompt,
            return_binary=return_binary,
            safe_mode=safe_mode,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            width=width
        )

        mock_async_client.post.assert_awaited_once_with(
            "image/generate",
            json_data={
                "model": model,
                "prompt": prompt,
                "cfg_scale": cfg_scale,
                "embed_exif_metadata": embed_exif_metadata,
                "format": format,
                "height": height,
                "hide_watermark": hide_watermark,
                "lora_strength": lora_strength,
                "negative_prompt": negative_prompt,
                "return_binary": return_binary,
                "safe_mode": safe_mode,
                "seed": seed,
                "steps": steps,
                "style_preset": style_preset,
                "width": width,
            },
            cast_to=ImageResponse
        )
        mock_async_client._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_return_binary(self, mock_async_client):
        """Test async generate method when return_binary is True."""
        image_resource = AsyncImage(mock_async_client)
        model = "test-model-async"
        prompt = "test prompt async"
        return_binary = True
        mock_async_client._request.return_value = b"binary image data async"

        result = await image_resource.generate(model=model, prompt=prompt, return_binary=return_binary)

        mock_async_client._request.assert_awaited_once_with(
            method="POST",
            path="image/generate",
            json_data={
                "model": model,
                "prompt": prompt,
                "return_binary": return_binary,
            },
            headers={"Accept": "image/*"},
            raw_response=True
        )
        mock_async_client.post.assert_not_called()
        assert result == b"binary image data async"

    @pytest.mark.asyncio
    async def test_simple_generate_with_required_params(self, mock_async_client):
        """Test async simple_generate method with only required parameters."""
        image_resource = AsyncImage(mock_async_client)
        model = "test-model-async"
        prompt = "test prompt async"

        await image_resource.simple_generate(model=model, prompt=prompt)

        mock_async_client.post.assert_awaited_once_with(
            "images/generations",
            json_data={
                "model": model,
                "prompt": prompt,
            },
            cast_to=SimpleImageResponse
        )

    @pytest.mark.asyncio
    async def test_simple_generate_with_all_optional_params(self, mock_async_client):
        """Test async simple_generate method with all optional parameters."""
        image_resource = AsyncImage(mock_async_client)
        model = "test-model-async"
        prompt = "test prompt async"
        background = "opaque"
        moderation = "auto"
        n = 1
        output_compression = 90
        output_format = "jpeg"
        quality = "standard"
        response_format = "b64_json"
        size = "1024x1024"
        style = "natural"
        user = "test-user-async"

        await image_resource.simple_generate(
            model=model,
            prompt=prompt,
            background=background,
            moderation=moderation,
            n=n,
            output_compression=output_compression,
            output_format=output_format,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
            user=user
        )

        mock_async_client.post.assert_awaited_once_with(
            "images/generations",
            json_data={
                "model": model,
                "prompt": prompt,
                "background": background,
                "moderation": moderation,
                "n": n,
                "output_compression": output_compression,
                "output_format": output_format,
                "quality": quality,
                "response_format": response_format,
                "size": size,
                "style": style,
                "user": user,
            },
            cast_to=SimpleImageResponse
        )

    @pytest.mark.asyncio
    @patch("venice_ai.resources.image.base64.b64encode")
    async def test_upscale_with_filepath(self, mock_b64encode, mock_async_client):
        """Test async upscale method with a file path."""
        image_resource = AsyncImage(mock_async_client)
        mock_b64encode.return_value = b"base64encodedimageasync"
        mock_async_client._request.return_value = b"upscaled image data async"

        # Use the dummy file from test data directory
        dummy_file_path = Path("e2e_tests/data/dummy_image_async.png")

        result = await image_resource.upscale(image=str(dummy_file_path))

        mock_async_client._request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data={"image": "base64encodedimageasync", "scale": 2.0},
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_async_client.post.assert_not_called()
        assert result == b"upscaled image data async"

    @pytest.mark.asyncio
    @patch("venice_ai.resources.image.base64.b64encode")
    async def test_upscale_with_bytes(self, mock_b64encode, mock_async_client):
        """Test async upscale method with bytes input."""
        image_resource = AsyncImage(mock_async_client)
        mock_b64encode.return_value = b"base64encodedimageasync"
        mock_async_client._request.return_value = b"upscaled image data async"

        image_bytes = b"image content bytes async"
        result = await image_resource.upscale(image=image_bytes)

        mock_async_client._request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data={"image": "base64encodedimageasync", "scale": 2.0},
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_async_client.post.assert_not_called()
        assert result == b"upscaled image data async"

    @pytest.mark.asyncio
    @patch("venice_ai.resources.image.base64.b64encode")
    async def test_upscale_with_file_like_object(self, mock_b64encode, mock_async_client):
        """Test async upscale method with a file-like object."""
        image_resource = AsyncImage(mock_async_client)
        mock_b64encode.return_value = b"base64encodedimageasync"
        mock_async_client._request.return_value = b"upscaled image data async"

        file_obj = io.BytesIO(b"file object content async")
        result = await image_resource.upscale(image=file_obj)

        mock_async_client._request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data={"image": "base64encodedimageasync", "scale": 2.0},
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_async_client.post.assert_not_called()
        assert result == b"upscaled image data async"

    @pytest.mark.asyncio
    @patch("venice_ai.resources.image.base64.b64encode")
    async def test_upscale_with_optional_params(self, mock_b64encode, mock_async_client):
        """Test async upscale method with optional parameters."""
        image_resource = AsyncImage(mock_async_client)
        mock_b64encode.return_value = b"base64encodedimageasync"
        mock_async_client._request.return_value = b"upscaled image data async"

        image_bytes = b"image content bytes async"
        enhance = False
        enhance_creativity = 0.9
        enhance_prompt = True
        replication = 0.7
        scale = 2.0

        result = await image_resource.upscale(
            image=image_bytes,
            enhance=enhance,
            enhance_creativity=enhance_creativity,
            enhance_prompt=enhance_prompt,
            replication=replication,
            scale=scale
        )

        mock_async_client._request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data={
                "image": "base64encodedimageasync",
                "enhance": False, # False is passed as a boolean
                "replication": replication,
                "scale": scale,
            },
            headers={"Accept": "application/json"},
            raw_response=True,
            timeout=None
        )
        mock_async_client.post.assert_not_called()
        assert result == b"upscaled image data async"

    @pytest.mark.asyncio
    async def test_list_styles(self, mock_async_client):
        """Test async list_styles method."""
        image_resource = AsyncImage(mock_async_client)
        mock_async_client.get.return_value = {"data": [{"id": "style2", "name": "Style 2"}]}

        result = await image_resource.list_styles()

        mock_async_client.get.assert_awaited_once_with("image/styles")
        assert result == {"data": [{"id": "style2", "name": "Style 2"}]}

class TestGuessImageType:
    def test_guess_image_type_jpeg(self):
        """Test _guess_image_type for jpeg."""
        assert _guess_image_type("image.jpg") == "jpeg"
        assert _guess_image_type("image.JPeG") == "jpeg"

    def test_guess_image_type_png(self):
        """Test _guess_image_type for png."""
        assert _guess_image_type("image.png") == "png"
        assert _guess_image_type("image.PNG") == "png"

    def test_guess_image_type_webp(self):
        """Test _guess_image_type for webp."""
        assert _guess_image_type("image.webp") == "webp"
        assert _guess_image_type("image.WEBP") == "webp"

    def test_guess_image_type_gif(self):
        """Test _guess_image_type for gif."""
        assert _guess_image_type("image.gif") == "gif"
        assert _guess_image_type("image.GIF") == "gif"

    def test_guess_image_type_unknown(self):
        """Test _guess_image_type for unknown extension."""
        assert _guess_image_type("image.txt") == "octet-stream"
        assert _guess_image_type("image_without_extension") == "octet-stream"
        assert _guess_image_type("image.tar.gz") == "octet-stream"