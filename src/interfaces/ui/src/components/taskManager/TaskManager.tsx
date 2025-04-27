import React from 'react';
import { Box, Typography, Paper, Container, Grid, Divider } from '@mui/material';

const TaskManager: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Paper 
          elevation={3} 
          sx={{ 
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            minHeight: '70vh'
          }}
        >
          <Typography variant="h4" component="h1" gutterBottom>
            タスク管理インターフェース
          </Typography>
          <Typography variant="h6" color="textSecondary" sx={{ mb: 4 }}>
            開発中 - タスク管理インターフェースは現在実装中です
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 2, 
                  height: '200px',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  進行中のタスク
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="body2" color="textSecondary">
                  現在進行中のタスクがここに表示されます
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 2, 
                  height: '200px',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  予定タスク
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="body2" color="textSecondary">
                  これから実行予定のタスクがここに表示されます
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 2, 
                  height: '200px',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  タスク詳細
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="body2" color="textSecondary">
                  選択したタスクの詳細情報がここに表示されます
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
};

export default TaskManager;
