using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using Unity.Sentis;
using HoloLab.DNN.Segmentation;
using System.Collections.Generic;

public class Segmentation : MonoBehaviour
{
    [SerializeField] private RawImage input_image;
    [SerializeField] private RawImage output_image;
    [SerializeField] private ModelAsset encoder_asset;
    [SerializeField] private ModelAsset decoder_asset;
    [SerializeField, Range(0.0f, 1.0f)] private float alpha = 0.5f;

    private SegmentationModel_MobileSAM model = null;
    private List<Color> colors;

    private void Start()
    {
        // Create Segmentation Model
        model = new SegmentationModel_MobileSAM(encoder_asset, decoder_asset, BackendType.GPUCompute);

        // Create Colors
        colors = new List<Color>() { Color.clear, new Color(1.0f, 0.0f, 0.0f, alpha) };
    }

    public void OnClick(BaseEventData event_data)
    {
        // Get Texture from Raw Image
        var input_texture = input_image.texture as Texture2D;
        if (input_texture == null)
        {
            return;
        }

        // Get Texture from Raw Image
        var pointer_data = event_data as PointerEventData;
        var rect_transform = input_image.transform as RectTransform;
        var click_point = GetClickPoint(pointer_data, rect_transform, input_texture.width, input_texture.height);

        // Segment Area
        var indices_texture = model.Segment(input_texture, click_point);

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

    private Vector2 GetClickPoint(PointerEventData pointer_data, RectTransform rect_transform, int width, int height)
    {
        Vector2 local_point;
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(rect_transform, pointer_data.position, null, out local_point))
        {
            return new Vector2(-1.0f, -1.0f);
        }
        local_point = local_point / rect_transform.rect.size + rect_transform.pivot;
        var click_point = new Vector2((int)((local_point.x * width) + 0.5f), (int)(((1.0f - local_point.y) * height) + 0.5f));
        return click_point;
    }

    private void OnDestroy()
    {
        model?.Dispose();
        model = null;
    }
}
