import React from 'react';
import { Box, Grid, Typography, Paper, useTheme } from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import ChatInterface from './components/ChatInterface';
import SystemStatus from './components/SystemStatus';
import ActivityFeed from './components/ActivityFeed';
import QuickActions from './components/QuickActions';
import KnowledgeStats from './components/KnowledgeStats';

const PageTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
  marginBottom: theme.spacing(3),
  color: theme.palette.primary.main,
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
  marginBottom: theme.spacing(2),
  color: theme.palette.mode === 'dark' 
    ? theme.palette.primary.light 
    : theme.palette.primary.main,
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  background: theme.palette.mode === 'dark' 
    ? 'rgba(66, 66, 66, 0.7)'
    : 'rgba(255, 255, 255, 0.9)',
  backdropFilter: 'blur(8px)',
  borderRadius: 16,
  overflow: 'hidden',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
  border: `1px solid ${theme.palette.divider}`,
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const MotionContainer = styled(motion.div)({
  height: '100%',
});

const Dashboard: React.FC = () => {
  const theme = useTheme();
  
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  };
  
  return (
    <MotionContainer
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <PageTitle variant="h4" component="h1">
        Command Center
      </PageTitle>
      
      <Grid container spacing={3} sx={{ height: 'calc(100% - 50px)' }}>
        {/* Main Chat Interface - Left 2/3 */}
        <Grid item xs={12} md={8} sx={{ height: { md: '100%' } }}>
          <motion.div variants={itemVariants} style={{ height: '100%' }}>
            <StyledPaper sx={{ p: 0 }}>
              <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                <SectionTitle variant="h6">
                  Jarviee Assistant
                </SectionTitle>
              </Box>
              <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
                <ChatInterface />
              </Box>
            </StyledPaper>
          </motion.div>
        </Grid>
        
        {/* Right Column - System Info and Activity */}
        <Grid item xs={12} md={4} container direction="column" spacing={3} sx={{ height: { md: '100%' } }}>
          {/* System Status */}
          <Grid item xs={12} sx={{ flexShrink: 0 }}>
            <motion.div variants={itemVariants}>
              <StyledPaper sx={{ p: 2 }}>
                <SectionTitle variant="h6">
                  System Status
                </SectionTitle>
                <SystemStatus />
              </StyledPaper>
            </motion.div>
          </Grid>
          
          {/* Quick Actions */}
          <Grid item xs={12} sx={{ flexShrink: 0 }}>
            <motion.div variants={itemVariants}>
              <StyledPaper sx={{ p: 2 }}>
                <SectionTitle variant="h6">
                  Quick Actions
                </SectionTitle>
                <QuickActions />
              </StyledPaper>
            </motion.div>
          </Grid>
          
          {/* Knowledge Stats */}
          <Grid item xs={12} sx={{ flexShrink: 0 }}>
            <motion.div variants={itemVariants}>
              <StyledPaper sx={{ p: 2 }}>
                <SectionTitle variant="h6">
                  Knowledge Stats
                </SectionTitle>
                <KnowledgeStats />
              </StyledPaper>
            </motion.div>
          </Grid>
          
          {/* Activity Feed */}
          <Grid item xs={12} sx={{ flexGrow: 1 }}>
            <motion.div variants={itemVariants} style={{ height: '100%' }}>
              <StyledPaper sx={{ p: 2 }}>
                <SectionTitle variant="h6">
                  Recent Activity
                </SectionTitle>
                <ActivityFeed />
              </StyledPaper>
            </motion.div>
          </Grid>
        </Grid>
      </Grid>
    </MotionContainer>
  );
};

export default Dashboard;
