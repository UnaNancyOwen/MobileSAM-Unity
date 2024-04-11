using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using HoloLab.DNN.Segmentation;

namespace Sample
{
    public class Segmentation : MonoBehaviour
    {
        [SerializeField] private RawImage input_image;
        [SerializeField] private RawImage output_image;
        [SerializeField] private ModelAsset encoder_asset;
        [SerializeField] private ModelAsset decoder_asset;
        [SerializeField, Range(0.0f, 1.0f)] private float alpha = 0.5f;

        private SegmentationModel_MobileSAM model = null;
        private Selector selector = null;
        private List<Color> colors;

        private void Start()
        {
            // Create Segmentation Model
            model = new SegmentationModel_MobileSAM(encoder_asset, decoder_asset, BackendType.GPUCompute);

            // Create Colors
            colors = new List<Color>() { Color.clear, new Color(1.0f, 0.0f, 0.0f, alpha) };

            // Create Selector
            var rect_transform = input_image.transform as RectTransform;
            var width = (input_image.texture as Texture2D).width;
            var height = (input_image.texture as Texture2D).height;
            selector = new Selector(rect_transform, width, height);
            selector.OnPointSelected += OnPointSelect;
            selector.OnRectSelected += OnRectSelect;
        }

        private void Update()
        {
            // Update Selector
            selector.Update();
        }

        public void OnPointSelect(object sender, PointEventArgs e)
        {
            // Get Texture from Raw Image
            var input_texture = input_image.texture as Texture2D;
            if (input_texture == null)
            {
                return;
            }

            // Segment Area using Point annotation
            var indices_texture = model.Segment(input_texture, e.point);

            // Draw Area on Unity UI
            var colorized_texture = Visualizer.ColorizeArea(indices_texture, colors);
            if (output_image.texture == null)
            {
                output_image.color = Color.white;
                output_image.texture = new Texture2D(indices_texture.width, indices_texture.height, TextureFormat.RGBA32, false);
            }
            Graphics.CopyTexture(colorized_texture, output_image.texture);

            // Destroy Texture
            Destroy(colorized_texture);
            Destroy(indices_texture);
        }

        public void OnRectSelect(object sender, RectEventArgs e)
        {
            // Get Texture from Raw Image
            var input_texture = input_image.texture as Texture2D;
            if (input_texture == null)
            {
                return;
            }

            // Segment Area using Bounding Box annotation
            var indices_texture = model.Segment(input_texture, e.rect);

            // Draw Area on Unity UI
            var colorized_texture = Visualizer.ColorizeArea(indices_texture, colors);
            if (output_image.texture == null)
            {
                output_image.color = Color.white;
                output_image.texture = new Texture2D(indices_texture.width, indices_texture.height, TextureFormat.RGBA32, false);
            }
            Graphics.CopyTexture(colorized_texture, output_image.texture);

            // Destroy Texture
            Destroy(colorized_texture);
            Destroy(indices_texture);
        }

        private void OnDestroy()
        {
            model?.Dispose();
            model = null;

            selector?.Dispose();
            selector = null;
        }
    }
}
