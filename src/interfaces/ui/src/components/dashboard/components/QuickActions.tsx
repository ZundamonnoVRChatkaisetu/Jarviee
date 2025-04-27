import React from 'react';
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  IconButton, 
  Tooltip,
  useTheme
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import AddIcon from '@mui/icons-material/Add';
import CodeIcon from '@mui/icons-material/Code';
import BoltIcon from '@mui/icons-material/Bolt';
import SearchIcon from '@mui/icons-material/Search';
import AssessmentIcon from '@mui/icons-material/Assessment';
import BugReportIcon from '@mui/icons-material/BugReport';
import DescriptionIcon from '@mui/icons-material/Description';
import { useNavigate } from 'react-router-dom';

const ActionGrid = styled(Grid)(({ theme }) => ({
  margin: theme.spacing(1, 0),
}));

const ActionButton = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1.5),
  textAlign: 'center',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(0, 0, 0, 0.2)' 
    : 'rgba(0, 0, 0, 0.03)',
  border: `1px solid ${theme.palette.divider}`,
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark' 
      ? 'rgba(100, 181, 246, 0.1)' 
      : 'rgba(25, 118, 210, 0.05)',
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[2],
    borderColor: theme.palette.primary.main,
  },
}));

const IconContainer = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(100, 181, 246, 0.1)' 
    : 'rgba(25, 118, 210, 0.05)',
  borderRadius: '50%',
  padding: theme.spacing(1),
  marginBottom: theme.spacing(1),
  color: theme.palette.primary.main,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const ActionLabel = styled(Typography)(({ theme }) => ({
  fontSize: '0.75rem',
  color: theme.palette.text.secondary,
  marginTop: theme.spacing(0.5),
}));

// Define actions
const actions = [
  {
    id: 'new-code',
    title: 'New Code',
    icon: <CodeIcon />,
    color: '#64B5F6',
    description: 'Create a new code file',
    path: '/code-editor',
  },
  {
    id: 'quick-task',
    title: 'New Task',
    icon: <BoltIcon />,
    color: '#FFB74D',
    description: 'Create a new task',
    path: '/task-manager',
  },
  {
    id: 'knowledge-search',
    title: 'Search',
    icon: <SearchIcon />,
    color: '#4DD0E1',
    description: 'Search knowledge base',
    path: '/knowledge-explorer',
  },
  {
    id: 'debug-code',
    title: 'Debug',
    icon: <BugReportIcon />,
    color: '#F06292',
    description: 'Debug existing code',
    path: '/code-editor?mode=debug',
  },
  {
    id: 'generate-report',
    title: 'Report',
    icon: <AssessmentIcon />,
    color: '#81C784',
    description: 'Generate a report',
    path: '/dashboard?action=report',
  },
  {
    id: 'new-doc',
    title: 'Document',
    icon: <DescriptionIcon />,
    color: '#9575CD',
    description: 'Create documentation',
    path: '/knowledge-explorer?action=new-doc',
  },
];

const QuickActions: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  
  const handleActionClick = (path: string) => {
    navigate(path);
  };
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 10, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  };
  
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <ActionGrid container spacing={2}>
        {actions.map((action) => (
          <Grid item xs={6} key={action.id}>
            <motion.div variants={itemVariants}>
              <Tooltip title={action.description} arrow>
                <ActionButton 
                  elevation={0}
                  onClick={() => handleActionClick(action.path)}
                >
                  <IconContainer sx={{ color: action.color }}>
                    {action.icon}
                  </IconContainer>
                  <Typography variant="body2" fontWeight={500}>
                    {action.title}
                  </Typography>
                  <ActionLabel variant="caption">
                    {action.description}
                  </ActionLabel>
                </ActionButton>
              </Tooltip>
            </motion.div>
          </Grid>
        ))}
      </ActionGrid>
      
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center',
          mt: 1
        }}
      >
        <Tooltip title="Add custom action">
          <IconButton
            size="small"
            sx={{
              border: `1px dashed ${theme.palette.divider}`,
              borderRadius: 1,
              p: 0.5
            }}
          >
            <AddIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </motion.div>
  );
};

export default QuickActions;
