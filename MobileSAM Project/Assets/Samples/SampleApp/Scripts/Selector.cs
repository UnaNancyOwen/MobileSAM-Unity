using System;
using UnityEngine;

namespace Sample
{
    /// <summary>
    /// point event args
    /// </summary>
    public class PointEventArgs : EventArgs
    {
        public Vector2 point;

        public PointEventArgs(Vector2 point)
        {
            this.point = point;
        }
    }

    /// <summary>
    /// rect event args
    /// </summary>
    public class RectEventArgs : EventArgs
    {
        public Rect rect;

        public RectEventArgs(Rect rect)
        {
            this.rect = rect;
        }
    }

    /// <summary>
    /// selector
    /// </summary>
    public class Selector : IDisposable
    {
        private RectTransform rect_transform = null;
        private Vector2 start_position, stop_position;
        private Vector2 click_position;
        private int width, height;

        private Vector3 mouse_drag_start_position = Vector3.zero;
        private float mouse_drag_distance = 0.0f;
        private float mouse_drag_threshold = 100.0f;
        private bool is_mouse_dragging = false;
        private bool is_mouse_button_downed = false;

        /// <summary>
        /// point selected event handler
        /// </summary>
        public event EventHandler<PointEventArgs> OnPointSelected = null;

        /// <summary>
        /// rect selected event handler
        /// </summary>
        public event EventHandler<RectEventArgs> OnRectSelected = null;

        /// <summary>
        /// create selector
        /// </summary>
        /// <param name="rect_transform">rect transform of target object</param>
        /// <param name="width">image width</param>
        /// <param name="height">image height</param>
        public Selector(RectTransform rect_transform, int width, int height)
        {
            this.rect_transform = rect_transform;
            this.width = width;
            this.height = height;
        }

        /// <summary>
        /// dispose selector
        /// </summary>
        public void Dispose()
        {
            OnPointSelected = null;
            OnRectSelected = null;
        }

        /// <summary>
        /// update selector
        /// </summary>
        public void Update()
        {
            var mouse_position = Input.mousePosition;

            if (Input.GetMouseButtonDown(0))
            {
                mouse_drag_start_position = mouse_position;
                is_mouse_button_downed = true;
                mouse_drag_distance = 0.0f;
            }

            if (is_mouse_button_downed)
            {
                mouse_drag_distance += (mouse_position - mouse_drag_start_position).magnitude;

                if (mouse_drag_threshold < mouse_drag_distance)
                {
                    if (!is_mouse_dragging)
                    {
                        start_position = GetMousePosition(mouse_drag_start_position, rect_transform, width, height);
                        is_mouse_dragging = true;
                    }
                }
            }

            if (Input.GetMouseButtonUp(0))
            {
                mouse_drag_start_position = Vector3.zero;
                is_mouse_button_downed = false;
                mouse_drag_distance = 0.0f;

                if (is_mouse_dragging)
                {
                    stop_position = GetMousePosition(mouse_position, rect_transform, width, height);
                    is_mouse_dragging = false;

                    if (IsContain(start_position, new Vector2(0.0f, 0.0f), new Vector2(width, height)) && IsContain(stop_position, new Vector2(0.0f, 0.0f), new Vector2(width, height)))
                    {
                        var diff = start_position - stop_position;
                        var size = new Vector2(Math.Abs(diff.x), Math.Abs(diff.y));
                        var positon = Vector2.Lerp(start_position, stop_position, 0.5f) - (size * 0.5f);
                        var rect = new Rect(positon, size);
                        OnRectSelected?.Invoke(this, new RectEventArgs(rect));
                    }
                }
                else
                {
                    click_position = GetMousePosition(mouse_position, rect_transform, width, height);

                    if (IsContain(click_position, new Vector2(0.0f, 0.0f), new Vector2(width, height)))
                    {
                        OnPointSelected?.Invoke(this, new PointEventArgs(click_position));
                    }
                }
            }
        }

        private Vector2 GetMousePosition(Vector3 position, RectTransform rect_transform, int width, int height)
        {
            Vector2 local_point;
            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(rect_transform, position, null, out local_point))
            {
                return new Vector2(-1.0f, -1.0f);
            }
            local_point = local_point / rect_transform.rect.size + rect_transform.pivot;
            var mouse_point = new Vector2((int)((local_point.x * width) + 0.5f), (int)(((1.0f - local_point.y) * height) + 0.5f));
            return mouse_point;
        }

        private static bool IsContain(Vector2 point, Vector2 min, Vector2 max)
        {
            return IsContain(point.x, min.x, max.x) && IsContain(point.y, min.y, max.y);
        }

        private static bool IsContain(float value, float min, float max)
        {
            return min <= value && value < max;
        }
    }
}
