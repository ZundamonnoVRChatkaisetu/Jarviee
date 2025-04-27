import React from 'react';
import { Box, Typography, Paper, Container } from '@mui/material';

const CodeEditor: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Paper 
          elevation={3} 
          sx={{ 
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minHeight: '70vh'
          }}
        >
          <Typography variant="h4" component="h1" gutterBottom>
            コードエディター
          </Typography>
          <Typography variant="h6" color="textSecondary" sx={{ mb: 4 }}>
            開発中 - コードエディター統合機能は現在実装中です
          </Typography>
          <Box 
            sx={{ 
              width: '100%', 
              height: '400px', 
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
              p: 2,
              fontFamily: 'monospace',
              fontSize: '0.9rem',
              overflow: 'auto'
            }}
          >
            <Typography variant="body2" component="pre" color="textSecondary">
              {'// コードエディター機能\n// 実装予定の機能:\n// - シンタックスハイライト\n// - コード補完\n// - エラーチェック\n// - コード実行\n// - デバッグ支援\n'}
            </Typography>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default CodeEditor;
