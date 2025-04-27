import React from 'react';
import { Box, Typography, Paper, Container, Divider, List, ListItem, ListItemIcon, ListItemText, Switch } from '@mui/material';
import { Settings as SettingsIcon, Colorize, Storage, Code, Security, Notifications } from '@mui/icons-material';

const Settings: React.FC = () => {
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
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <SettingsIcon sx={{ mr: 1 }} />
            <Typography variant="h4" component="h1">
              設定
            </Typography>
          </Box>
          <Typography variant="h6" color="textSecondary" sx={{ mb: 4 }}>
            開発中 - 設定画面は現在実装中です
          </Typography>
          
          <Paper elevation={1} sx={{ mb: 3 }}>
            <List>
              <ListItem>
                <ListItemIcon>
                  <Colorize />
                </ListItemIcon>
                <ListItemText primary="ダークモード" secondary="ダークテーマを有効にする" />
                <Switch edge="end" />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <Storage />
                </ListItemIcon>
                <ListItemText primary="知識ベース自動更新" secondary="新しい情報を自動的に取得" />
                <Switch edge="end" />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <Code />
                </ListItemIcon>
                <ListItemText primary="コードエディタ設定" secondary="コードエディタの動作をカスタマイズ" />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <Security />
                </ListItemIcon>
                <ListItemText primary="プライバシー設定" secondary="データ共有と保護の設定" />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <Notifications />
                </ListItemIcon>
                <ListItemText primary="通知設定" secondary="通知の頻度とタイプを設定" />
              </ListItem>
            </List>
          </Paper>
          
          <Typography variant="body2" color="textSecondary" align="center">
            Jarviee バージョン 0.1.0
          </Typography>
        </Paper>
      </Box>
    </Container>
  );
};

export default Settings;
