import cv2
import time


PRIMARY_PINK  = (210, 130, 255)   # brighter violet-pink
ACCENT_PINK   = (170,  90, 240)   # slightly deeper accent
SOFT_PINK     = (230, 170, 255)   # pale pink for secondary text
WHITE         = (255, 255, 255)
LIGHT_GRAY    = (235, 230, 245)   # brighter near-white
OVERLAY_DARK  = (45,  28,  58)    # rich dark purple (not pure black)
GREEN_BADGE   = ( 80, 185, 110)   # hand-detected green
MUTED         = (110,  95, 125)   # muted dots / no-hand



def _rounded_rect_overlay(img, x1, y1, x2, y2, radius, color, alpha):
    """Internal: blend a filled rounded rect onto img in-place."""
    overlay = img.copy()
   
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _text_size(text, scale, thickness=1):
    (w, h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h, baseline


def draw_text(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_pill(img, text, x, y, bg_color, text_color=WHITE,
              font_scale=0.45, thickness=1):
    """Draw pill badge; returns (x2, y2) so caller knows the bounds."""
    tw, th, baseline = _text_size(text, font_scale, thickness)
    px, py = 10, 5
    x2 = x + tw + 2 * px
    y2 = y + th + 2 * py + baseline
    r  = (y2 - y) // 2
    _rounded_rect_overlay(img, x, y, x2, y2, r, bg_color, alpha=0.92)
    draw_text(img, text, x + px, y2 - py - baseline // 2,
              font_scale, text_color, thickness)
    return x2, y2




class ResponsiveUI:
    """
    All dimensions derived from frame size so everything fits at any resolution.
    Layout regions:
      ┌───────────────── top_bar ──────────────────┐
      │ left_panel │                               │
      │ stab_dots  │     (camera feed)             │
      │            │                               │
      └───────────────── bottom_bar ───────────────┘
    """

    def __init__(self, frame):
        self.frame = frame
        self.H, self.W, _ = frame.shape

  
        self.m   = max(8, int(0.012 * self.W))   # outer margin
        self.gap = max(6, int(0.008 * self.W))   # inner gap

     
        self.cw  = min(self.W - 2 * self.m, 1100)
        self.cx  = (self.W - self.cw) // 2       # content left edge


        self.top_h  = max(44, int(0.082 * self.H))
        self.bot_h  = max(52, int(0.110 * self.H))

   
        self.panel_y  = self.m + self.top_h + self.gap
        self.panel_w  = max(140, int(0.20 * self.cw))
        avail_h       = self.H - self.panel_y - self.bot_h - 2 * self.m - self.gap
        self.panel_h  = max(130, min(int(0.30 * self.H), avail_h - 36))
     
        self.dots_y   = self.panel_y + self.panel_h + self.gap

     
        self.fs_title  = max(0.55, self.W / 1600)   # top-bar title
        self.fs_hint   = max(0.38, self.W / 2400)   # keyboard hint
        self.fs_label  = max(0.38, self.W / 2200)   # small labels
        self.fs_conf   = max(0.42, self.W / 2000)   # "conf XX%"
        self.fs_letter = max(2.0,  self.W / 480)    # big letter
        self.fs_word   = max(0.85, self.W / 900)    # bottom word

        self.bar_h     = max(6, int(0.010 * self.H))

    def top_bar(self, hand_count=0):
        x1 = self.cx
        y1 = self.m
        x2 = self.cx + self.cw
        y2 = y1 + self.top_h
        _rounded_rect_overlay(self.frame, x1, y1, x2, y2, 12, OVERLAY_DARK, 0.72)

     
        baseline_y = y1 + int(self.top_h * 0.64)
        draw_text(self.frame, "Sign Language Detector",
                  x1 + 16, baseline_y, self.fs_title, PRIMARY_PINK, 1)

   
        hint = "C=clear    S=space    Q=quit"
        tw, th, _ = _text_size(hint, self.fs_hint)
        hint_x = x2 - tw - 16
        hint_y = baseline_y
        draw_text(self.frame, hint, hint_x, hint_y, self.fs_hint, LIGHT_GRAY, 1)

     
        badge_text  = (f"{hand_count} hand{'s' if hand_count != 1 else ''} detected"
                       if hand_count > 0 else "no hand detected")
        badge_color = GREEN_BADGE if hand_count > 0 else MUTED
        bw, _,_ = _text_size(badge_text, 0.40)
        badge_x = x2 - bw - 36
        draw_pill(self.frame, badge_text,
                  badge_x, y1 + 4,
                  badge_color, WHITE, font_scale=0.40)

  
    def left_panel(self, letter, confidence, stable_count, stable_max):
        px  = self.cx
        py  = self.panel_y
        pw  = self.panel_w
        ph  = self.panel_h

   
        _rounded_rect_overlay(self.frame, px, py, px + pw, py + ph,
                               12, OVERLAY_DARK, 0.72)

        inner_x = px + 12

    
        draw_text(self.frame, "detecting",
                  inner_x, py + 20, self.fs_label, SOFT_PINK)

   
        lw, lh, _ = _text_size(letter, self.fs_letter, 2)
        letter_x   = px + (pw - lw) // 2
        letter_y   = py + int(ph * 0.68)
        draw_text(self.frame, letter, letter_x, letter_y,
                  self.fs_letter, PRIMARY_PINK, 2)

        conf_label_y = py + int(ph * 0.83)
        draw_text(self.frame, f"conf  {int(confidence * 100)}%",
                  inner_x, conf_label_y, self.fs_conf, LIGHT_GRAY)

       
        bar_y = conf_label_y + 8
        bar_w = pw - 24
        _rounded_rect_overlay(self.frame,
                               inner_x, bar_y,
                               inner_x + bar_w, bar_y + self.bar_h,
                               self.bar_h // 2, (70, 55, 85), 0.80)
        fill = max(0, int(bar_w * min(confidence, 1.0)))
        if fill > self.bar_h:
            _rounded_rect_overlay(self.frame,
                                   inner_x, bar_y,
                                   inner_x + fill, bar_y + self.bar_h,
                                   self.bar_h // 2, PRIMARY_PINK, 0.95)

        dot_box_h = 28
        dby = self.dots_y
        _rounded_rect_overlay(self.frame,
                               px, dby, px + pw, dby + dot_box_h,
                               10, OVERLAY_DARK, 0.68)

  
        draw_text(self.frame, "stability",
                  inner_x, dby + dot_box_h - 8, self.fs_label, SOFT_PINK)

        dot_r       = max(4, int(pw * 0.040))
        total_dots_w = stable_max * (dot_r * 2 + 4) - 4
        dot_start_x  = px + pw - total_dots_w - 12
        dot_cy       = dby + dot_box_h // 2
        for i in range(stable_max):
            dcx   = dot_start_x + i * (dot_r * 2 + 4) + dot_r
            color = PRIMARY_PINK if i < stable_count else MUTED
            cv2.circle(self.frame, (dcx, dot_cy), dot_r, color, -1, cv2.LINE_AA)


    def bottom_bar(self, word, spoken):
        bh  = self.bot_h
        y1  = self.H - bh - self.m
        y2  = self.H - self.m
        cx  = self.cx
        cw  = self.cw

        _rounded_rect_overlay(self.frame, cx, y1, cx + cw, y2,
                               12, OVERLAY_DARK, 0.72)

        inner_x = cx + 16

   
        draw_text(self.frame, "word",
                  inner_x, y1 + 18, self.fs_label, SOFT_PINK)

   
        word_display = word.strip() if word.strip() else "_"
        _, wh, _ = _text_size(word_display, self.fs_word, 2)
        word_y = y2 - max(12, int(bh * 0.18))
        draw_text(self.frame, word_display,
                  inner_x, word_y, self.fs_word, WHITE, 2)

        if spoken:
            bw, _,_ = _text_size("spoken", 0.40)
            badge_x = cx + cw - bw - 36
            badge_y = y1 + (bh - 24) // 2
            draw_pill(self.frame, "spoken",
                      badge_x, badge_y, ACCENT_PINK, WHITE, 0.40)