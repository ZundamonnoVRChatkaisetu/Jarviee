import React from 'react';
import { Outlet } from 'react-router-dom';
import { Box, useTheme } from '@mui/material';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import Header from '../components/common/Header';
import Sidebar from '../components/common/Sidebar';
import ContextPanel from '../components/common/ContextPanel';
import NotificationSystem from '../components/common/NotificationSystem';

const MainLayout: React.FC = () => {
  const theme = useTheme();
  const { sidebarOpen, contextPanelOpen } = useSelector((state: RootState) => state.ui);
  
  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <Header />
      
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pt: `var(--header-height)`,
          ml: { xs: 0, md: sidebarOpen ? 'var(--sidebar-width)' : 0 },
          mr: { xs: 0, lg: contextPanelOpen ? 'var(--context-panel-width)' : 0 },
          transition: theme.transitions.create(['margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          height: '100%',
          overflow: 'auto',
          position: 'relative',
          backgroundColor: 'background.default',
        }}
      >
        {/* Background Elements */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 'var(--z-background)',
            pointerEvents: 'none',
            overflow: 'hidden',
          }}
        >
          {/* Radial gradient background */}
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              width: '120%',
              height: '120%',
              transform: 'translate(-50%, -50%)',
              background: 'radial-gradient(circle, rgba(100,181,246,0.1) 0%, rgba(18,18,18,0) 70%)',
            }}
          />
          
          {/* Grid lines */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundImage: 'linear-gradient(rgba(100,181,246,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(100,181,246,0.05) 1px, transparent 1px)',
              backgroundSize: '50px 50px',
              opacity: 0.5,
            }}
          />
        </Box>
        
        {/* Page Content */}
        <Box
          sx={{
            height: '100%',
            position: 'relative',
            zIndex: 'var(--z-base)',
            padding: 3,
          }}
        >
          <Outlet />
        </Box>
      </Box>
      
      {/* Context Panel */}
      <ContextPanel />
      
      {/* Notification System */}
      <NotificationSystem />
    </Box>
  );
};

export default MainLayout;
