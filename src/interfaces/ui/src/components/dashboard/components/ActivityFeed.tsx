import React from 'react';
import { 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon,
  Divider,
  Avatar,
  Chip,
  useTheme
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CodeIcon from '@mui/icons-material/Code';
import StorageIcon from '@mui/icons-material/Storage';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import BuildIcon from '@mui/icons-material/Build';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import FolderIcon from '@mui/icons-material/Folder';
import { motion } from 'framer-motion';

const ActivityContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const ActivityList = styled(List)(({ theme }) => ({
  width: '100%',
  overflowY: 'auto',
  padding: 0,
  flexGrow: 1,
  '& .MuiListItem-root': {
    paddingTop: theme.spacing(1.5),
    paddingBottom: theme.spacing(1.5),
  },
}));

const ActivityIcon = styled(Avatar)(({ theme }) => ({
  width: 36,
  height: 36,
  backgroundColor: theme.palette.primary.main,
}));

const TimeChip = styled(Chip)(({ theme }) => ({
  height: 24,
  fontSize: '0.75rem',
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(0, 0, 0, 0.3)' 
    : 'rgba(0, 0, 0, 0.08)',
  color: theme.palette.text.secondary,
}));

// Mock activity data for demonstration
const activityData = [
  {
    id: '1',
    type: 'code',
    title: 'Code optimization completed',
    description: 'Python algorithm efficiency improved by 85%',
    timestamp: Date.now() - 1000 * 60 * 5, // 5 minutes ago
    icon: <CodeIcon />,
  },
  {
    id: '2',
    type: 'knowledge',
    title: 'Knowledge base updated',
    description: 'Added 15 new entries on Machine Learning',
    timestamp: Date.now() - 1000 * 60 * 30, // 30 minutes ago
    icon: <StorageIcon />,
  },
  {
    id: '3',
    type: 'task',
    title: 'Scheduled task completed',
    description: 'Data backup and verification successful',
    timestamp: Date.now() - 1000 * 60 * 60, // 1 hour ago
    icon: <BuildIcon />,
  },
  {
    id: '4',
    type: 'learn',
    title: 'Learning session completed',
    description: 'New information on TypeScript 5.0 features',
    timestamp: Date.now() - 1000 * 60 * 60 * 3, // 3 hours ago
    icon: <MenuBookIcon />,
  },
  {
    id: '5',
    type: 'system',
    title: 'System update installed',
    description: 'Core components updated to version 0.9.2',
    timestamp: Date.now() - 1000 * 60 * 60 * 12, // 12 hours ago
    icon: <SmartToyIcon />,
  },
  {
    id: '6',
    type: 'file',
    title: 'Project structure reorganized',
    description: 'Optimized folder hierarchy for better navigation',
    timestamp: Date.now() - 1000 * 60 * 60 * 24, // 1 day ago
    icon: <FolderIcon />,
  },
];

const getRelativeTime = (timestamp: number): string => {
  const now = Date.now();
  const diffMs = now - timestamp;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffDay > 0) {
    return diffDay === 1 ? '1 day ago' : `${diffDay} days ago`;
  }
  if (diffHour > 0) {
    return diffHour === 1 ? '1 hour ago' : `${diffHour} hours ago`;
  }
  if (diffMin > 0) {
    return diffMin === 1 ? '1 minute ago' : `${diffMin} minutes ago`;
  }
  return 'Just now';
};

const getIconColor = (type: string) => {
  switch (type) {
    case 'code':
      return '#64B5F6'; // blue
    case 'knowledge':
      return '#4DD0E1'; // teal
    case 'task':
      return '#FFB74D'; // orange
    case 'learn':
      return '#81C784'; // green
    case 'system':
      return '#9575CD'; // purple
    case 'file':
      return '#4FC3F7'; // light blue
    default:
      return '#64B5F6';
  }
};

const ActivityFeed: React.FC = () => {
  const theme = useTheme();
  
  return (
    <ActivityContainer>
      <ActivityList>
        {activityData.map((activity, index) => (
          <React.Fragment key={activity.id}>
            <ListItem
              alignItems="flex-start"
              component={motion.li}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                <ActivityIcon 
                  sx={{ bgcolor: getIconColor(activity.type) }}
                >
                  {activity.icon}
                </ActivityIcon>
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" fontWeight={500}>
                      {activity.title}
                    </Typography>
                    <TimeChip
                      label={getRelativeTime(activity.timestamp)}
                      size="small"
                    />
                  </Box>
                }
                secondary={
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    {activity.description}
                  </Typography>
                }
              />
            </ListItem>
            {index < activityData.length - 1 && (
              <Divider variant="inset" component="li" />
            )}
          </React.Fragment>
        ))}
      </ActivityList>
    </ActivityContainer>
  );
};

export default ActivityFeed;
