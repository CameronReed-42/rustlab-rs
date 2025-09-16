use crate::plot::types::Color;

#[derive(Debug, Clone)]
pub struct Theme {
    pub background_color: Color,
    pub grid_color: Color,
    pub text_color: Color,
    pub palette: Vec<Color>,
    pub name: String,
}

impl Theme {
    pub fn default_light() -> Self {
        Self {
            background_color: Color { r: 255, g: 255, b: 255 }, // White
            grid_color: Color { r: 200, g: 200, b: 200 },       // Light gray
            text_color: Color { r: 0, g: 0, b: 0 },             // Black
            palette: vec![
                Color::BLUE,
                Color::RED,
                Color::GREEN,
                Color::ORANGE,
                Color::PURPLE,
                Color::BLACK,
            ],
            name: "Light".to_string(),
        }
    }

    pub fn dark() -> Self {
        Self {
            background_color: Color { r: 33, g: 37, b: 41 },    // Dark gray
            grid_color: Color { r: 100, g: 100, b: 100 },       // Medium gray
            text_color: Color { r: 255, g: 255, b: 255 },       // White
            palette: vec![
                Color { r: 70, g: 130, b: 180 },  // Steel blue
                Color { r: 220, g: 20, b: 60 },   // Crimson
                Color { r: 50, g: 205, b: 50 },   // Lime green
                Color { r: 255, g: 140, b: 0 },   // Dark orange
                Color { r: 138, g: 43, b: 226 },  // Blue violet
                Color { r: 255, g: 255, b: 255 }, // White
            ],
            name: "Dark".to_string(),
        }
    }

    pub fn scientific() -> Self {
        Self {
            background_color: Color { r: 248, g: 248, b: 255 }, // Ghost white
            grid_color: Color { r: 211, g: 211, b: 211 },       // Light gray
            text_color: Color { r: 25, g: 25, b: 112 },         // Midnight blue
            palette: vec![
                Color { r: 0, g: 0, b: 139 },     // Dark blue
                Color { r: 139, g: 0, b: 0 },     // Dark red
                Color { r: 0, g: 100, b: 0 },     // Dark green
                Color { r: 255, g: 165, b: 0 },   // Orange
                Color { r: 75, g: 0, b: 130 },    // Indigo
                Color { r: 139, g: 69, b: 19 },   // Saddle brown
            ],
            name: "Scientific".to_string(),
        }
    }

    pub fn colorblind_friendly() -> Self {
        Self {
            background_color: Color { r: 255, g: 255, b: 255 }, // White
            grid_color: Color { r: 200, g: 200, b: 200 },       // Light gray
            text_color: Color { r: 0, g: 0, b: 0 },             // Black
            palette: vec![
                Color { r: 0, g: 114, b: 178 },   // Blue
                Color { r: 213, g: 94, b: 0 },    // Vermillion
                Color { r: 0, g: 158, b: 115 },   // Bluish green
                Color { r: 204, g: 121, b: 167 }, // Reddish purple
                Color { r: 86, g: 180, b: 233 },  // Sky blue
                Color { r: 230, g: 159, b: 0 },   // Orange
            ],
            name: "Colorblind Friendly".to_string(),
        }
    }

    pub fn get_color(&self, index: usize) -> Color {
        self.palette[index % self.palette.len()]
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::default_light()
    }
}