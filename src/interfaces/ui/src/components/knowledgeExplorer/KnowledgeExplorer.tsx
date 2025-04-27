import React from 'react';
import { Box, Typography, Paper, Container, Grid } from '@mui/material';

const KnowledgeExplorer: React.FC = () => {
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
            知識探索インターフェース
          </Typography>
          <Typography variant="h6" color="textSecondary" sx={{ mb: 4 }}>
            開発中 - 知識探索インターフェースは現在実装中です
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 2, 
                  height: '400px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  知識カテゴリ
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  カテゴリツリーが表示されます
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={8}>
              <Paper 
                elevation={2} 
                sx={{ 
                  p: 2, 
                  height: '400px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  知識コンテンツ
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  選択した知識の詳細が表示されます
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
};

export default KnowledgeExplorer;
