import React from 'react';
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  Divider,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  useTheme,
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { setContextPanelOpen } from '../../store/slices/uiSlice';
import CloseIcon from '@mui/icons-material/Close';
import InfoIcon from '@mui/icons-material/Info';
import HistoryIcon from '@mui/icons-material/History';
import DescriptionIcon from '@mui/icons-material/Description';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import MemoryIcon from '@mui/icons-material/Memory';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';

const StyledDrawer = styled(Drawer)(({ theme }) => ({
  width: 'var(--context-panel-width)',
  flexShrink: 0,
  '& .MuiDrawer-paper': {
    width: 'var(--context-panel-width)',
    boxSizing: 'border-box',
    top: 'var(--header-height)',
    height: `calc(100% - var(--header-height))`,
    background: theme.palette.mode === 'dark' 
      ? 'rgba(33, 33, 33, 0.9)'
      : 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(8px)',
    borderLeft: `1px solid ${theme.palette.divider}`,
  },
}));

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`contextpanel-tabpanel-${index}`}
      aria-labelledby={`contextpanel-tab-${index}`}
      style={{ height: '100%', overflow: 'auto' }}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 2, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const a11yProps = (index: number) => {
  return {
    id: `contextpanel-tab-${index}`,
    'aria-controls': `contextpanel-tabpanel-${index}`,
  };
};

// Mock data for demonstration
const contextData = {
  knowledge: [
    { id: 1, title: 'Python Programming', type: 'language', confidence: 0.95 },
    { id: 2, title: 'React Framework', type: 'framework', confidence: 0.87 },
    { id: 3, title: 'Machine Learning', type: 'domain', confidence: 0.78 },
  ],
  history: [
    { id: 1, title: 'Code optimization request', timestamp: Date.now() - 3600000 },
    { id: 2, title: 'Knowledge graph exploration', timestamp: Date.now() - 7200000 },
    { id: 3, title: 'Task scheduling', timestamp: Date.now() - 86400000 },
  ],
  resources: [
    { id: 1, title: 'Python Documentation', type: 'documentation', relevance: 0.92 },
    { id: 2, title: 'React Best Practices', type: 'article', relevance: 0.85 },
    { id: 3, title: 'ML Model Examples', type: 'repository', relevance: 0.78 },
  ],
  metrics: {
    cpu: 42,
    memory: 65,
    tasks: 3,
    uptime: '3h 24m',
  }
};

const ContextPanel: React.FC = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const { contextPanelOpen } = useSelector((state: RootState) => state.ui);
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleClose = () => {
    dispatch(setContextPanelOpen(false));
  };

  return (
    <StyledDrawer
      variant="persistent"
      anchor="right"
      open={contextPanelOpen}
    >
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        p: 2 
      }}>
        <Typography 
          variant="h6" 
          component="div"
          sx={{ 
            fontSize: '1rem',
            fontWeight: 500,
            color: 'primary.main',
          }}
        >
          Context Panel
        </Typography>
        <IconButton onClick={handleClose} size="small">
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
      
      <Divider />
      
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        height: 'calc(100% - 58px)' // Subtract header height
      }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="context panel tabs"
          variant="fullWidth"
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            '& .MuiTabs-indicator': {
              height: 3,
            },
          }}
        >
          <Tab 
            icon={<InfoIcon fontSize="small" />} 
            label="Context" 
            {...a11yProps(0)} 
            sx={{ minHeight: 48 }}
          />
          <Tab 
            icon={<HistoryIcon fontSize="small" />} 
            label="History" 
            {...a11yProps(1)}
            sx={{ minHeight: 48 }}
          />
          <Tab 
            icon={<DescriptionIcon fontSize="small" />} 
            label="Resources" 
            {...a11yProps(2)}
            sx={{ minHeight: 48 }}
          />
          <Tab 
            icon={<TrendingUpIcon fontSize="small" />} 
            label="Metrics" 
            {...a11yProps(3)}
            sx={{ minHeight: 48 }}
          />
        </Tabs>
        
        <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
          <TabPanel value={tabValue} index={0}>
            <Typography variant="subtitle2" gutterBottom>
              Current Context
            </Typography>
            <List>
              {contextData.knowledge.map((item) => (
                <ListItem key={item.id} sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <InfoIcon fontSize="small" color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.title}
                    secondary={`Type: ${item.type}`}
                  />
                  <Chip 
                    label={`${Math.round(item.confidence * 100)}%`}
                    size="small"
                    color={item.confidence > 0.9 ? 'success' : item.confidence > 0.7 ? 'primary' : 'default'}
                  />
                </ListItem>
              ))}
            </List>
          </TabPanel>
          
          <TabPanel value={tabValue} index={1}>
            <Typography variant="subtitle2" gutterBottom>
              Interaction History
            </Typography>
            <List>
              {contextData.history.map((item) => (
                <ListItem key={item.id} sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <HistoryIcon fontSize="small" color="action" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.title}
                    secondary={new Date(item.timestamp).toLocaleString()}
                    primaryTypographyProps={{ variant: 'body2' }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                </ListItem>
              ))}
            </List>
          </TabPanel>
          
          <TabPanel value={tabValue} index={2}>
            <Typography variant="subtitle2" gutterBottom>
              Relevant Resources
            </Typography>
            <List>
              {contextData.resources.map((item) => (
                <ListItem key={item.id} sx={{ px: 0 }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <DescriptionIcon fontSize="small" color="info" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.title}
                    secondary={`Type: ${item.type}`}
                    primaryTypographyProps={{ variant: 'body2' }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                  <Chip 
                    label={`${Math.round(item.relevance * 100)}%`}
                    size="small"
                    color={item.relevance > 0.9 ? 'success' : item.relevance > 0.7 ? 'info' : 'default'}
                  />
                </ListItem>
              ))}
            </List>
          </TabPanel>
          
          <TabPanel value={tabValue} index={3}>
            <Typography variant="subtitle2" gutterBottom>
              System Metrics
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                CPU Usage
              </Typography>
              <Box sx={{ 
                height: 8, 
                bgcolor: 'background.paper',
                borderRadius: 4,
                overflow: 'hidden'
              }}>
                <Box 
                  component={motion.div}
                  initial={{ width: 0 }}
                  animate={{ width: `${contextData.metrics.cpu}%` }}
                  transition={{ duration: 1 }}
                  sx={{ 
                    height: '100%', 
                    bgcolor: contextData.metrics.cpu > 80 ? 'error.main' : 
                             contextData.metrics.cpu > 60 ? 'warning.main' : 
                             'success.main',
                  }}
                />
              </Box>
              <Typography variant="caption" color="text.secondary">
                {contextData.metrics.cpu}%
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Memory Usage
              </Typography>
              <Box sx={{ 
                height: 8, 
                bgcolor: 'background.paper',
                borderRadius: 4,
                overflow: 'hidden'
              }}>
                <Box 
                  component={motion.div}
                  initial={{ width: 0 }}
                  animate={{ width: `${contextData.metrics.memory}%` }}
                  transition={{ duration: 1 }}
                  sx={{ 
                    height: '100%', 
                    bgcolor: contextData.metrics.memory > 80 ? 'error.main' : 
                             contextData.metrics.memory > 60 ? 'warning.main' : 
                             'success.main',
                  }}
                />
              </Box>
              <Typography variant="caption" color="text.secondary">
                {contextData.metrics.memory}%
              </Typography>
            </Box>
            
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.05)',
              p: 2,
              borderRadius: 2,
              mt: 3
            }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Active Tasks
                </Typography>
                <Typography variant="h6" color="primary">
                  {contextData.metrics.tasks}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Uptime
                </Typography>
                <Typography variant="h6" color="primary">
                  {contextData.metrics.uptime}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Status
                </Typography>
                <Typography variant="h6" color="success.main">
                  Active
                </Typography>
              </Box>
            </Box>
            
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Chip 
                icon={<MemoryIcon fontSize="small" />}
                label="System Operational" 
                color="success"
                variant="outlined"
              />
            </Box>
          </TabPanel>
        </Box>
      </Box>
    </StyledDrawer>
  );
};

export default ContextPanel;
