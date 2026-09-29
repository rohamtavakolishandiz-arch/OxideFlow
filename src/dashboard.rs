use crossterm::{
    event::{self, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, Paragraph},
    Terminal,
};
use std::io::{stdout, Stdout};
use std::time::Duration;

pub struct Dashboard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Dashboard {
    pub fn new() -> Self {
        enable_raw_mode().expect("Failed to enable raw mode");
        stdout().execute(EnterAlternateScreen).expect("Failed to enter alternate screen");
        let backend = CrosstermBackend::new(stdout());
        let terminal = Terminal::new(backend).expect("Failed to create terminal");
        Self { terminal }
    }

    pub fn draw(&mut self, episode: usize, score: u32, epsilon: f64, memory_len: usize, memory_cap: usize) {
        let _ = self.terminal.draw(|f| {
            // Split the terminal into 4 vertical rows
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Min(0),
                ].as_ref())
                .split(f.area());

            // ROW 1: Title
            let title = Paragraph::new(format!(" 🚀 OxideFlow Max-Speed Headless Training | Episode: {} ", episode))
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(title, chunks[0]);

            // ROW 2: Live Metrics
            let score_text = Paragraph::new(format!(" Current Score: {} | Exploration Rate: {:.2}%", score, epsilon * 100.0))
                .block(Block::default().borders(Borders::ALL).title(" Performance Metrics "));
            f.render_widget(score_text, chunks[1]);

            // ROW 3: Replay Buffer Gauge
            let mem_pct = (memory_len as f64 / memory_cap as f64).clamp(0.0, 1.0);
            let mem_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" Replay Buffer Memory "))
                .gauge_style(Style::default().fg(Color::Yellow))
                .ratio(mem_pct);
            f.render_widget(mem_gauge, chunks[2]);

            // ROW 4: Footer
            let help_text = Paragraph::new(" Press 'q' to safely save the checkpoint and exit.")
                .style(Style::default().fg(Color::DarkGray));
            f.render_widget(help_text, chunks[3]);
        });
    }

    // Listens for terminal keypresses (since Winit has no window to listen to)
    pub fn check_quit(&self) -> bool {
        if let Ok(true) = event::poll(Duration::from_millis(0)) {
            if let Ok(event::Event::Key(key)) = event::read() {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Char('Q') {
                    return true;
                }
            }
        }
        false
    }
}

// Automatically cleans up the terminal if the program crashes or quits
impl Drop for Dashboard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = stdout().execute(LeaveAlternateScreen);
    }
}